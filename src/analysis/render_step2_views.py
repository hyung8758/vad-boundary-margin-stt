import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import ensure_dir, get_split_group, load_config, setup_logging
from src.metrics import compute_word_csid


LOGGER = logging.getLogger("render_step2_views")

CONDITION_ORDER = ["baseline", "vad_default", "vad_fa_informed"]
CONDITION_COLORS = {
    "baseline": "#4c78a8",
    "vad_default": "#f58518",
    "vad_fa_informed": "#54a24b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render step2 VAD comparison tables and figures.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step2.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def condition_label(condition, policy_tag):
    if condition == "baseline":
        return "baseline"
    if condition == "silero_vad_default":
        return "vad_default"
    if condition == policy_tag:
        return "vad_fa_informed"
    return condition


def add_word_csid_columns(frame):
    metric_rows = []
    for row in frame.itertuples():
        metric_rows.append(compute_word_csid(row.normalized_reference, row.normalized_hypothesis))
    csid = pd.DataFrame(metric_rows)
    return pd.concat([frame.reset_index(drop=True), csid], axis=1)


def add_baseline_delta_columns(frame):
    baseline = frame[frame["condition"] == "baseline"][
        [
            "source_utterance_id",
            "split",
            "cer",
            "first_word_error_rate",
            "last_word_error_rate",
            "word_s",
            "word_d",
            "word_i",
        ]
    ].drop_duplicates(subset=["source_utterance_id", "split"])

    baseline = baseline.rename(
        columns={
            "cer": "baseline_cer",
            "first_word_error_rate": "baseline_first_word_error_rate",
            "last_word_error_rate": "baseline_last_word_error_rate",
            "word_s": "baseline_word_s",
            "word_d": "baseline_word_d",
            "word_i": "baseline_word_i",
        }
    )
    merged = frame.merge(baseline, on=["source_utterance_id", "split"], how="left")
    merged["delta_cer"] = merged["cer"] - merged["baseline_cer"]
    merged["delta_first_word_error_rate"] = merged["first_word_error_rate"] - merged["baseline_first_word_error_rate"]
    merged["delta_last_word_error_rate"] = merged["last_word_error_rate"] - merged["baseline_last_word_error_rate"]
    merged["delta_word_s"] = merged["word_s"] - merged["baseline_word_s"]
    merged["delta_word_d"] = merged["word_d"] - merged["baseline_word_d"]
    merged["delta_word_i"] = merged["word_i"] - merged["baseline_word_i"]
    return merged


def format_percent(series):
    return (series * 100.0).round(2)


def save_figure(fig, output_path):
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote figure %s", pdf_path)


def build_condition_table(frame, group_columns):
    summary = (
        frame.groupby(group_columns, dropna=False)[
            [
                "cer",
                "delta_cer",
                "changed_from_baseline",
                "first_word_error_rate",
                "last_word_error_rate",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = frame.groupby(group_columns, dropna=False).size().reset_index(name="n_examples")
    summary = summary.merge(counts, on=group_columns, how="left")
    summary["cer_mean_pct"] = format_percent(summary["cer"])
    summary["delta_cer_mean_pct"] = format_percent(summary["delta_cer"])
    summary["changed_rate_pct"] = format_percent(summary["changed_from_baseline"])
    summary["first_word_error_rate_pct"] = format_percent(summary["first_word_error_rate"])
    summary["last_word_error_rate_pct"] = format_percent(summary["last_word_error_rate"])
    return summary[
        group_columns
        + [
            "n_examples",
            "cer_mean_pct",
            "delta_cer_mean_pct",
            "changed_rate_pct",
            "first_word_error_rate_pct",
            "last_word_error_rate_pct",
        ]
    ]


def build_csid_table(frame):
    summary = (
        frame.groupby(["split_group", "condition"], dropna=False)[
            ["word_s", "word_d", "word_i", "delta_word_s", "delta_word_d", "delta_word_i"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary.round(4)


def build_cer_boundary_changed_table(clean_other_table):
    table = clean_other_table[
        [
            "split_group",
            "condition_label",
            "n_examples",
            "cer_mean_pct",
            "first_word_error_rate_pct",
            "last_word_error_rate_pct",
            "changed_rate_pct",
        ]
    ].copy()
    return table.sort_values(["split_group", "condition_label"]).reset_index(drop=True)


def render_metric_bar(summary, output_path, metric_column, title, ylabel, condition_order):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    split_groups = ["clean", "other"]

    for axis, split_group in zip(axes, split_groups):
        rows = summary[summary["split_group"] == split_group].copy()
        rows["condition_label"] = pd.Categorical(rows["condition_label"], categories=condition_order, ordered=True)
        rows = rows.sort_values("condition_label")
        x_values = list(range(len(rows)))
        bar_colors = [CONDITION_COLORS.get(value, "#999999") for value in rows["condition_label"]]
        axis.bar(x_values, rows[metric_column], color=bar_colors, width=0.65)
        axis.set_xticks(x_values)
        axis.set_xticklabels(rows["condition_label"], rotation=15)
        axis.set_title(split_group)
        axis.set_xlabel("Condition")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="y", labelleft=True)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(title)
    save_figure(fig, output_path)


def render_step2_sid_plot(frame, output_path):
    metrics = [
        ("delta_word_s", "Delta S"),
        ("delta_word_i", "Delta I"),
        ("delta_word_d", "Delta D"),
    ]

    summary = (
        frame.groupby(["split_group", "condition_label"], dropna=False)[["delta_word_s", "delta_word_i", "delta_word_d"]]
        .mean(numeric_only=True)
        .reset_index()
    )

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for row_index, (metric_name, metric_label) in enumerate(metrics):
        for column_index, split_group in enumerate(["clean", "other"]):
            axis = axes[row_index][column_index]
            rows = summary[summary["split_group"] == split_group].copy()
            rows["condition_label"] = pd.Categorical(rows["condition_label"], categories=CONDITION_ORDER, ordered=True)
            rows = rows.sort_values("condition_label")
            x_values = list(range(len(rows)))
            bar_colors = [CONDITION_COLORS.get(value, "#999999") for value in rows["condition_label"]]
            axis.bar(x_values, rows[metric_name], color=bar_colors, width=0.65)
            axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
            axis.set_xticks(x_values)
            axis.set_xticklabels(rows["condition_label"], rotation=15)
            axis.set_title(f"{split_group} / {metric_label}")
            axis.set_xlabel("Condition")
            axis.set_ylabel(metric_label)
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Step2 SID Change vs Baseline")
    save_figure(fig, output_path)


def render_step2_boundary_error_plot(clean_other_table, output_path):
    metrics = [
        ("first_word_error_rate_pct", "First-Word Error Rate (%)"),
        ("last_word_error_rate_pct", "Last-Word Error Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for row_index, (metric_name, metric_label) in enumerate(metrics):
        for column_index, split_group in enumerate(["clean", "other"]):
            axis = axes[row_index][column_index]
            rows = clean_other_table[clean_other_table["split_group"] == split_group].copy()
            rows["condition_label"] = pd.Categorical(rows["condition_label"], categories=CONDITION_ORDER, ordered=True)
            rows = rows.sort_values("condition_label")
            x_values = list(range(len(rows)))
            bar_colors = [CONDITION_COLORS.get(value, "#999999") for value in rows["condition_label"]]
            axis.bar(x_values, rows[metric_name], color=bar_colors, width=0.65)
            axis.set_xticks(x_values)
            axis.set_xticklabels(rows["condition_label"], rotation=15)
            axis.set_title(f"{split_group} / {metric_label}")
            axis.set_xlabel("Condition")
            axis.set_ylabel(metric_label)
            axis.tick_params(axis="y", labelleft=True)
            axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Step2 First / Last Word Error")
    save_figure(fig, output_path)


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START render_step2_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, args.output_dir)

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = Path(args.output_dir) if args.output_dir else exp_root / "analysis"
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    eval_dir = exp_root / "eval"
    paths = sorted(eval_dir.glob(f"{args.dataset}_*_per_utterance.csv"))
    if not paths:
        raise FileNotFoundError(f"No step2 per-utterance CSV files found for dataset={args.dataset}")

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    policy_tag = f"silero_vad_{config['vad_policy']['policy_name']}"
    valid_conditions = {"baseline", "silero_vad_default", policy_tag}
    frame = frame[frame["condition"].isin(valid_conditions)].copy()
    frame["split_group"] = frame["split"].map(get_split_group)
    frame = add_word_csid_columns(frame)
    frame = add_baseline_delta_columns(frame)
    frame["condition_label"] = frame["condition"].map(lambda value: condition_label(value, policy_tag))

    condition_table = build_condition_table(frame, ["split", "split_group", "condition", "condition_label"])
    clean_other_table = build_condition_table(frame, ["split_group", "condition", "condition_label"])
    csid_table = build_csid_table(frame)
    cer_boundary_changed_table = build_cer_boundary_changed_table(clean_other_table)

    condition_table.to_csv(tables_dir / "condition_table.csv", index=False)
    clean_other_table.to_csv(tables_dir / "clean_other_table.csv", index=False)
    csid_table.to_csv(tables_dir / "csid_table.csv", index=False)
    cer_boundary_changed_table.to_csv(tables_dir / "cer_boundary_changed_table.csv", index=False)
    LOGGER.info("Wrote table %s", tables_dir / "condition_table.csv")
    LOGGER.info("Wrote table %s", tables_dir / "clean_other_table.csv")
    LOGGER.info("Wrote table %s", tables_dir / "csid_table.csv")
    LOGGER.info("Wrote table %s", tables_dir / "cer_boundary_changed_table.csv")

    render_metric_bar(clean_other_table, figures_dir / "vad_cer", "cer_mean_pct", "Step2 VAD Comparison (CER)", "CER (%)", CONDITION_ORDER)
    render_step2_sid_plot(frame, figures_dir / "vad_delta_sid")
    render_step2_boundary_error_plot(clean_other_table, figures_dir / "vad_first_last_word_error")

    LOGGER.info("END render_step2_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, output_dir)


if __name__ == "__main__":
    main()
