import argparse
import csv
import logging
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import ensure_dir, get_split_group, load_config, setup_logging
from src.metrics import compute_word_csid


LOGGER = logging.getLogger("render_step1_views")

SELECTED_CONDITIONS = [
    "baseline",
    "leading_m100",
    "leading_p000",
    "leading_p300",
    "trailing_m100",
    "trailing_p000",
    "trailing_p300",
]

METRIC_FIELDS = [
    "cer",
    "delta_cer",
    "first_word_match_rate",
    "last_word_match_rate",
    "delta_first_word_error_rate",
    "delta_last_word_error_rate",
    "changed_from_baseline",
    "word_c",
    "word_s",
    "word_d",
    "word_i",
    "delta_word_c",
    "delta_word_s",
    "delta_word_d",
    "delta_word_i",
]

MAX_MARGIN_MS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Render paper-ready tables and figures from per-utterance eval CSV files.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--dataset", required=True, help="Dataset prefix used in exp/eval filenames.")
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step1.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: exp/<exp-subdir>/analysis")
    return parser.parse_args()


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value):
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def parse_bool(value):
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def condition_sort_key(condition):
    order = {name: index for index, name in enumerate(SELECTED_CONDITIONS)}
    if condition in order:
        return (0, order[condition], condition)
    return (1, 0, condition)


def mean_value(values):
    if not values:
        return None
    return statistics.fmean(values)


def std_value(values):
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def format_float(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def format_percent(value):
    if value is None:
        return ""
    return f"{value * 100.0:.2f}"


def load_per_utterance_rows(exp_root, dataset_name):
    paths = sorted((Path(exp_root) / "eval").glob(f"{dataset_name}_*_per_utterance.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-utterance CSV files found for dataset={dataset_name}")

    rows = []
    for path in paths:
        rows.extend(read_csv_rows(path))
    return rows


def prepare_rows(raw_rows):
    baseline_metrics = {}
    for row in raw_rows:
        normalized_reference = row.get("normalized_reference", row["reference"])
        normalized_hypothesis = row.get("normalized_hypothesis", row["hypothesis"])
        csid = compute_word_csid(normalized_reference, normalized_hypothesis)
        row["word_c"] = csid["word_c"]
        row["word_s"] = csid["word_s"]
        row["word_d"] = csid["word_d"]
        row["word_i"] = csid["word_i"]

        if row["experiment_type"] == "baseline":
            key = (row["split"], row["source_utterance_id"])
            baseline_metrics[key] = {
                "cer": parse_float(row["cer"]),
                "first_word_error_rate": parse_float(row["first_word_error_rate"]),
                "last_word_error_rate": parse_float(row["last_word_error_rate"]),
                "word_c": row["word_c"],
                "word_s": row["word_s"],
                "word_d": row["word_d"],
                "word_i": row["word_i"],
            }

    prepared = []
    for row in raw_rows:
        numeric_row = dict(row)
        numeric_row["cer"] = parse_float(row["cer"])
        numeric_row["first_word_match_rate"] = parse_float(row["first_word_match_rate"])
        numeric_row["last_word_match_rate"] = parse_float(row["last_word_match_rate"])
        numeric_row["first_word_error_rate"] = parse_float(row["first_word_error_rate"])
        numeric_row["last_word_error_rate"] = parse_float(row["last_word_error_rate"])
        numeric_row["changed_from_baseline"] = parse_bool(row["changed_from_baseline"])
        numeric_row["word_c"] = row["word_c"]
        numeric_row["word_s"] = row["word_s"]
        numeric_row["word_d"] = row["word_d"]
        numeric_row["word_i"] = row["word_i"]
        numeric_row["split_group"] = get_split_group(row["split"])

        key = (row["split"], row["source_utterance_id"])
        baseline = baseline_metrics.get(key)
        if baseline is None:
            numeric_row["delta_cer"] = None
            numeric_row["delta_first_word_error_rate"] = None
            numeric_row["delta_last_word_error_rate"] = None
            numeric_row["delta_word_c"] = None
            numeric_row["delta_word_s"] = None
            numeric_row["delta_word_d"] = None
            numeric_row["delta_word_i"] = None
        else:
            numeric_row["delta_cer"] = numeric_row["cer"] - baseline["cer"]
            numeric_row["delta_first_word_error_rate"] = numeric_row["first_word_error_rate"] - baseline["first_word_error_rate"]
            numeric_row["delta_last_word_error_rate"] = numeric_row["last_word_error_rate"] - baseline["last_word_error_rate"]
            numeric_row["delta_word_c"] = numeric_row["word_c"] - baseline["word_c"]
            numeric_row["delta_word_s"] = numeric_row["word_s"] - baseline["word_s"]
            numeric_row["delta_word_d"] = numeric_row["word_d"] - baseline["word_d"]
            numeric_row["delta_word_i"] = numeric_row["word_i"] - baseline["word_i"]
        prepared.append(numeric_row)
    return prepared


def aggregate_rows(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    output = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        record = {}
        for index, field in enumerate(group_fields):
            record[field] = key[index]
        record["n_examples"] = len(group_rows)
        for metric in METRIC_FIELDS:
            values = [row[metric] for row in group_rows if row[metric] is not None]
            record[f"{metric}_mean"] = mean_value(values)
            record[f"{metric}_std"] = std_value(values)
        output.append(record)
    return output


def parse_margin_condition(condition):
    if not condition.startswith(("leading_", "trailing_")):
        return None
    suffix = condition.split("_", 1)[1]
    sign = suffix[0]
    value = int(suffix[1:])
    if sign == "m":
        return -value
    return value


def is_valid_margin_ms(margin_ms):
    if margin_ms is None:
        return False
    return margin_ms <= MAX_MARGIN_MS


def format_margin_ticks(x_values):
    return [value for value in x_values if value % 100 == 0]


def collect_margin_rows(rows):
    margin_rows = []
    for row in rows:
        if row["experiment_type"] not in ("leading", "trailing"):
            continue
        margin_ms = parse_margin_condition(row["condition"])
        if not is_valid_margin_ms(margin_ms):
            continue
        row_copy = dict(row)
        row_copy["margin_ms"] = margin_ms
        margin_rows.append(row_copy)
    return margin_rows


def collect_pause_rows(rows):
    pause_rows = []
    for row in rows:
        if row["experiment_type"] != "pause_cut":
            continue
        parsed = parse_pause_condition(row["condition"])
        if parsed is None:
            continue
        bucket_label, ratio = parsed
        row_copy = dict(row)
        row_copy["pause_bucket"] = bucket_label
        row_copy["pause_ratio"] = ratio
        pause_rows.append(row_copy)
    return pause_rows


def parse_pause_condition(condition):
    if not condition.startswith("pause_"):
        return None
    parts = condition.split("_")
    bucket_start = parts[1][1:]
    bucket_end = parts[2]
    ratio = int(parts[3][1:]) / 100.0
    if bucket_end == "inf":
        bucket_label = f"{bucket_start}+"
    else:
        bucket_label = f"{bucket_start}-{bucket_end}"
    return bucket_label, ratio


def build_main_condition_table(rows):
    selected = [row for row in rows if row["condition"] in SELECTED_CONDITIONS]
    aggregated = aggregate_rows(selected, ["split", "condition"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "condition": row["condition"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "changed_rate_pct": format_percent(row["changed_from_baseline_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], condition_sort_key(row["condition"])))
    return formatted


def build_clean_other_table(rows):
    selected = [row for row in rows if row["condition"] in SELECTED_CONDITIONS]
    aggregated = aggregate_rows(selected, ["split_group", "condition"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split_group": row["split_group"],
                "condition": row["condition"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "cer_std_pct": format_percent(row["cer_std"]),
                "delta_cer_mean_pct": format_percent(row["delta_cer_mean"]),
                "delta_cer_std_pct": format_percent(row["delta_cer_std"]),
                "changed_rate_pct": format_percent(row["changed_from_baseline_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split_group"], condition_sort_key(row["condition"])))
    return formatted


def build_pause_table(rows):
    pause_rows = collect_pause_rows(rows)
    aggregated = aggregate_rows(pause_rows, ["split", "split_group", "pause_bucket", "pause_ratio"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "split_group": row["split_group"],
                "pause_bucket": row["pause_bucket"],
                "pause_ratio": row["pause_ratio"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "delta_cer_mean_pct": format_percent(row["delta_cer_mean"]),
                "delta_cer_std_pct": format_percent(row["delta_cer_std"]),
                "changed_rate_pct": format_percent(row["changed_from_baseline_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], row["pause_bucket"], row["pause_ratio"]))
    return formatted


def build_padding_csid_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["split", "split_group", "experiment_type", "condition", "margin_ms"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "split_group": row["split_group"],
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "margin_ms": row["margin_ms"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "delta_cer_mean_pct": format_percent(row["delta_cer_mean"]),
                "word_c_mean": format_float(row["word_c_mean"]),
                "word_s_mean": format_float(row["word_s_mean"]),
                "word_d_mean": format_float(row["word_d_mean"]),
                "word_i_mean": format_float(row["word_i_mean"]),
                "delta_word_c_mean": format_float(row["delta_word_c_mean"]),
                "delta_word_s_mean": format_float(row["delta_word_s_mean"]),
                "delta_word_d_mean": format_float(row["delta_word_d_mean"]),
                "delta_word_i_mean": format_float(row["delta_word_i_mean"]),
                "first_word_match_pct": format_percent(row["first_word_match_rate_mean"]),
                "last_word_match_pct": format_percent(row["last_word_match_rate_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], row["experiment_type"], float(row["margin_ms"])))
    return formatted


def build_leading_trailing_cer_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["split_group", "experiment_type", "condition", "margin_ms"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split_group": row["split_group"],
                "experiment_type": row["experiment_type"],
                "margin_ms": row["margin_ms"],
                "condition": row["condition"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "cer_std_pct": format_percent(row["cer_std"]),
            }
        )
    formatted.sort(key=lambda row: (row["split_group"], row["experiment_type"], float(row["margin_ms"])))
    return formatted


def build_leading_trailing_combined_error_profile_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["experiment_type", "condition", "margin_ms"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "margin_ms": row["margin_ms"],
                "n_examples": row["n_examples"],
                "word_s_mean": format_float(row["word_s_mean"]),
                "word_s_std": format_float(row["word_s_std"]),
                "word_i_mean": format_float(row["word_i_mean"]),
                "word_i_std": format_float(row["word_i_std"]),
                "word_d_mean": format_float(row["word_d_mean"]),
                "word_d_std": format_float(row["word_d_std"]),
                "first_word_error_rate_pct": format_percent(1.0 - row["first_word_match_rate_mean"]),
                "first_word_error_rate_std_pct": format_percent(row["first_word_match_rate_std"]),
                "last_word_error_rate_pct": format_percent(1.0 - row["last_word_match_rate_mean"]),
                "last_word_error_rate_std_pct": format_percent(row["last_word_match_rate_std"]),
            }
        )
    formatted.sort(key=lambda row: (row["experiment_type"], float(row["margin_ms"])))
    return formatted


def build_leading_trailing_first_last_error_rate_profile_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["experiment_type", "condition", "margin_ms"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "margin_ms": row["margin_ms"],
                "n_examples": row["n_examples"],
                "first_word_error_rate_pct": format_percent(1.0 - row["first_word_match_rate_mean"]),
                "first_word_error_rate_std_pct": format_percent(row["first_word_match_rate_std"]),
                "last_word_error_rate_pct": format_percent(1.0 - row["last_word_match_rate_mean"]),
                "last_word_error_rate_std_pct": format_percent(row["last_word_match_rate_std"]),
            }
        )
    formatted.sort(key=lambda row: (row["experiment_type"], float(row["margin_ms"])))
    return formatted


def build_boundary_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["split", "split_group", "experiment_type", "condition"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "split_group": row["split_group"],
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "n_examples": row["n_examples"],
                "first_word_match_pct": format_percent(row["first_word_match_rate_mean"]),
                "last_word_match_pct": format_percent(row["last_word_match_rate_mean"]),
                "delta_first_word_error_pct": format_percent(row["delta_first_word_error_rate_mean"]),
                "delta_last_word_error_pct": format_percent(row["delta_last_word_error_rate_mean"]),
                "word_s_mean": format_float(row["word_s_mean"]),
                "word_d_mean": format_float(row["word_d_mean"]),
                "word_i_mean": format_float(row["word_i_mean"]),
                "delta_word_s_mean": format_float(row["delta_word_s_mean"]),
                "delta_word_d_mean": format_float(row["delta_word_d_mean"]),
                "delta_word_i_mean": format_float(row["delta_word_i_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], row["experiment_type"], row["condition"]))
    return formatted


def build_pause_csid_table(rows):
    pause_rows = collect_pause_rows(rows)
    aggregated = aggregate_rows(pause_rows, ["split", "split_group", "pause_bucket", "pause_ratio"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "split_group": row["split_group"],
                "pause_bucket": row["pause_bucket"],
                "pause_ratio": row["pause_ratio"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "delta_cer_mean_pct": format_percent(row["delta_cer_mean"]),
                "word_c_mean": format_float(row["word_c_mean"]),
                "word_s_mean": format_float(row["word_s_mean"]),
                "word_d_mean": format_float(row["word_d_mean"]),
                "word_i_mean": format_float(row["word_i_mean"]),
                "delta_word_c_mean": format_float(row["delta_word_c_mean"]),
                "delta_word_s_mean": format_float(row["delta_word_s_mean"]),
                "delta_word_d_mean": format_float(row["delta_word_d_mean"]),
                "delta_word_i_mean": format_float(row["delta_word_i_mean"]),
                "changed_rate_pct": format_percent(row["changed_from_baseline_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], row["pause_bucket"], row["pause_ratio"]))
    return formatted


def build_best_margin_table(rows):
    selected = collect_margin_rows(rows)
    aggregated = aggregate_rows(selected, ["split_group", "experiment_type", "condition", "margin_ms"])
    grouped = defaultdict(list)
    for row in aggregated:
        grouped[(row["split_group"], row["experiment_type"])].append(row)

    formatted = []
    for key in sorted(grouped):
        split_group, experiment_type = key
        candidates = sorted(grouped[key], key=lambda row: (row["cer_mean"], row["margin_ms"]))
        best = candidates[0]
        formatted.append(
            {
                "split_group": split_group,
                "experiment_type": experiment_type,
                "best_condition": best["condition"],
                "best_margin_ms": best["margin_ms"],
                "cer_mean_pct": format_percent(best["cer_mean"]),
                "delta_cer_mean_pct": format_percent(best["delta_cer_mean"]),
                "changed_rate_pct": format_percent(best["changed_from_baseline_mean"]),
            }
        )
    return formatted


def build_all_condition_table(rows):
    aggregated = aggregate_rows(collect_margin_rows(rows), ["split", "experiment_type", "condition"])
    formatted = []
    for row in aggregated:
        formatted.append(
            {
                "split": row["split"],
                "experiment_type": row["experiment_type"],
                "condition": row["condition"],
                "n_examples": row["n_examples"],
                "cer_mean_pct": format_percent(row["cer_mean"]),
                "cer_std_pct": format_percent(row["cer_std"]),
                "delta_cer_mean_pct": format_percent(row["delta_cer_mean"]),
                "delta_cer_std_pct": format_percent(row["delta_cer_std"]),
                "changed_rate_pct": format_percent(row["changed_from_baseline_mean"]),
                "first_word_match_pct": format_percent(row["first_word_match_rate_mean"]),
                "last_word_match_pct": format_percent(row["last_word_match_rate_mean"]),
            }
        )
    formatted.sort(key=lambda row: (row["split"], row["experiment_type"], row["condition"]))
    return formatted


def save_table(output_dir, name, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / f"{name}.csv"
    write_csv(csv_path, fieldnames, rows)
    LOGGER.info("Wrote table %s", csv_path)


def save_figure(fig, output_dir, name):
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote figure %s", pdf_path)


def load_step2_baseline_cer_lines(exp_root):
    step2_table_path = exp_root.parent / "step2" / "analysis" / "tables" / "clean_other_table.csv"
    if not step2_table_path.exists():
        return {}

    lines = {}
    for row in read_csv_rows(step2_table_path):
        if row.get("condition_label") != "baseline":
            continue
        lines[row["split_group"]] = float(row["cer_mean_pct"])
    return lines


def render_margin_metric_plot(rows, output_dir, metric_name, figure_name, title, ylabel, baseline_lines=None):
    plot_rows = collect_margin_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    fig.subplots_adjust(top=0.82, wspace=0.24)
    title_y = 0.97
    panel_titles = {
        "clean": "Clean (dev+test)",
        "other": "Other (dev+test)",
    }

    for axis, split_group in zip(axes, ["clean", "other"]):
        x_ticks = []
        for experiment_type, color in [("leading", "#d1495b"), ("trailing", "#2b59c3")]:
            grouped = defaultdict(list)
            for row in plot_rows:
                if row["split_group"] != split_group or row["experiment_type"] != experiment_type:
                    continue
                grouped[row["margin_ms"]].append(row[metric_name])

            x_values = sorted(grouped)
            x_ticks = x_values
            y_values = [((mean_value(grouped[x]) or 0.0) * 100.0) for x in x_values]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=experiment_type.capitalize(), color=color)

        if baseline_lines and split_group in baseline_lines:
            axis.axhline(
                baseline_lines[split_group],
                color="#222222",
                linewidth=1.2,
                linestyle="--",
                label="Baseline",
            )
        axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        axis.set_title(panel_titles[split_group], fontsize=13, fontweight="bold")
        axis.set_xlabel("Margin (ms)", fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        axis.set_xticks(format_margin_ticks(x_ticks))
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=11, labelleft=True)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=title_y)
    save_figure(fig, output_dir, figure_name)


def render_step1_combined_error_profile(rows, output_dir):
    plot_rows = collect_margin_rows(rows)
    sid_metrics = [
        ("word_s", "S", "#d1495b"),
        ("word_i", "I", "#edae49"),
        ("word_d", "D", "#2b59c3"),
    ]
    boundary_metrics = [
        ("first_word_error_rate", "First", "#2a9d8f"),
        ("last_word_error_rate", "Last", "#7c4dff"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.25)
    experiment_types = [("leading", "Leading"), ("trailing", "Trailing")]
    sid_row_max = 0.0
    boundary_row_max = 0.0
    for column_index, (experiment_type, experiment_label) in enumerate(experiment_types):
        sid_axis = axes[0][column_index]
        sid_xticks = []
        for metric_name, label, color in sid_metrics:
            grouped = defaultdict(list)
            for row in plot_rows:
                if row["experiment_type"] != experiment_type:
                    continue
                grouped[row["margin_ms"]].append(row[metric_name])

            x_values = sorted(grouped)
            sid_xticks = x_values
            y_values = [mean_value(grouped[x]) or 0.0 for x in x_values]
            if y_values:
                sid_row_max = max(sid_row_max, max(y_values))
            sid_axis.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)

        sid_axis.set_title(f"{experiment_label} / SID")
        sid_axis.set_xlabel("Margin (ms)")
        sid_axis.set_ylabel("Mean Count per Utterance")
        sid_axis.set_xticks(format_margin_ticks(sid_xticks))
        sid_axis.tick_params(axis="x", labelbottom=True)
        sid_axis.tick_params(axis="y", labelsize=9)
        sid_axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        sid_axis.tick_params(axis="y", labelleft=True)
        sid_axis.grid(alpha=0.25)

        boundary_axis = axes[1][column_index]
        boundary_xticks = []
        for metric_name, label, color in boundary_metrics:
            grouped = defaultdict(list)
            for row in plot_rows:
                if row["experiment_type"] != experiment_type:
                    continue
                grouped[row["margin_ms"]].append(row[metric_name])

            x_values = sorted(grouped)
            boundary_xticks = x_values
            y_values = [((mean_value(grouped[x]) or 0.0) * 100.0) for x in x_values]
            if y_values:
                boundary_row_max = max(boundary_row_max, max(y_values))
            boundary_axis.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)

        boundary_axis.set_title(f"{experiment_label} / First-Last")
        boundary_axis.set_xlabel("Margin (ms)")
        boundary_axis.set_ylabel("Error Rate (%)")
        boundary_axis.set_xticks(format_margin_ticks(boundary_xticks))
        boundary_axis.tick_params(axis="x")
        boundary_axis.tick_params(axis="y", labelsize=9)
        boundary_axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        boundary_axis.tick_params(axis="y", labelleft=True)
        boundary_axis.grid(alpha=0.25)

    sid_upper = sid_row_max * 1.08 if sid_row_max > 0 else 1.0
    boundary_upper = boundary_row_max * 1.08 if boundary_row_max > 0 else 1.0
    for axis in axes[0]:
        axis.set_ylim(0.0, sid_upper)
    for axis in axes[1]:
        axis.set_ylim(0.0, boundary_upper)

    axes[0][1].legend()
    axes[1][1].legend()
    fig.suptitle("Leading / Trailing Error Profile (Clean + Other Combined)")
    save_figure(fig, output_dir, "leading_trailing_combined_error_profile")


def render_step1_first_last_error_rate_profile(rows, output_dir):
    plot_rows = collect_margin_rows(rows)
    metrics = [
        ("first_word_error_rate", "First-word", "#2a9d8f"),
        ("last_word_error_rate", "Last-word", "#7c4dff"),
    ]

    baseline_rows = [row for row in rows if row["experiment_type"] == "baseline"]
    baseline_first_values = [row["first_word_error_rate"] for row in baseline_rows if row["first_word_error_rate"] is not None]
    baseline_last_values = [row["last_word_error_rate"] for row in baseline_rows if row["last_word_error_rate"] is not None]
    baseline_first_error = (mean_value(baseline_first_values) or 0.0) * 100.0
    baseline_last_error = (mean_value(baseline_last_values) or 0.0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    fig.subplots_adjust(top=0.82, wspace=0.25)
    experiment_types = [("leading", "Leading"), ("trailing", "Trailing")]
    row_max = 0.0

    for column_index, (experiment_type, experiment_label) in enumerate(experiment_types):
        axis = axes[column_index]
        x_ticks = []
        for metric_name, label, color in metrics:
            grouped = defaultdict(list)
            for row in plot_rows:
                if row["experiment_type"] != experiment_type:
                    continue
                grouped[row["margin_ms"]].append(row[metric_name])

            x_values = sorted(grouped)
            x_ticks = x_values
            y_values = [((mean_value(grouped[x]) or 0.0) * 100.0) for x in x_values]
            if y_values:
                row_max = max(row_max, max(y_values))
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)

        axis.axhline(
            baseline_first_error,
            color="#2a9d8f",
            linewidth=1.1,
            linestyle="--",
            label="Baseline First-word",
            zorder=5,
        )
        axis.axhline(
            baseline_last_error,
            color="#7c4dff",
            linewidth=1.1,
            linestyle="--",
            label="Baseline Last-word",
            zorder=5,
        )
        axis.set_title(f"{experiment_label}", fontsize=13, fontweight="bold")
        axis.set_xlabel("Margin (ms)", fontsize=12, fontweight="bold")
        axis.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
        axis.set_xticks(format_margin_ticks(x_ticks))
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=11, labelleft=True)
        axis.axvline(0, color="#d1495b", linewidth=1.2, linestyle="--", alpha=0.8)
        axis.grid(alpha=0.25)

    upper = row_max * 1.08 if row_max > 0 else 1.0
    for axis in axes:
        axis.set_ylim(0.0, max(upper, baseline_first_error * 1.08, baseline_last_error * 1.08))

    axes[1].legend(fontsize=11)
    fig.suptitle(
        "First- and Last-Word Error under Leading and Trailing Margin Conditions",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    save_figure(fig, output_dir, "leading_trailing_first_last_error_rate_profile")


def main():
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START render_step1_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, args.output_dir)

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = Path(args.output_dir) if args.output_dir else exp_root / "analysis"
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")

    raw_rows = load_per_utterance_rows(exp_root, args.dataset)
    rows = prepare_rows(raw_rows)
    step2_baseline_cer_lines = load_step2_baseline_cer_lines(exp_root)

    save_table(tables_dir, "main_condition_table", build_main_condition_table(rows))
    save_table(tables_dir, "clean_other_condition_table", build_clean_other_table(rows))
    save_table(tables_dir, "pause_cut_table", build_pause_table(rows))
    save_table(tables_dir, "all_condition_stats", build_all_condition_table(rows))
    save_table(tables_dir, "padding_csid_table", build_padding_csid_table(rows))
    save_table(tables_dir, "leading_trailing_cer", build_leading_trailing_cer_table(rows))
    save_table(tables_dir, "leading_trailing_combined_error_profile", build_leading_trailing_combined_error_profile_table(rows))
    save_table(tables_dir, "leading_trailing_first_last_error_rate_profile", build_leading_trailing_first_last_error_rate_profile_table(rows))
    save_table(tables_dir, "boundary_condition_table", build_boundary_table(rows))
    save_table(tables_dir, "pause_cut_csid_table", build_pause_csid_table(rows))
    save_table(tables_dir, "best_margin_table", build_best_margin_table(rows))

    render_margin_metric_plot(
        rows,
        figures_dir,
        "cer",
        "leading_trailing_cer",
        "Effect of Leading and Trailing Margins on CER",
        "CER (%)",
        baseline_lines=step2_baseline_cer_lines,
    )
    render_step1_combined_error_profile(rows, figures_dir)
    render_step1_first_last_error_rate_profile(rows, figures_dir)
    LOGGER.info("END render_step1_views | dataset=%s exp_subdir=%s output_dir=%s", args.dataset, args.exp_subdir, output_dir)


if __name__ == "__main__":
    main()
