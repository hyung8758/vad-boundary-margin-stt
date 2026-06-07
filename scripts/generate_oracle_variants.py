import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.io import read_audio, write_audio
from src.audio.oracle_segments import (
    GeneratedVariant,
    generate_leading_variant,
    generate_pause_variants,
    generate_trailing_variant,
)
from src.common import ensure_dir, load_config, read_jsonl, setup_logging, write_jsonl
from src.data.textgrid_parser import parse_textgrid


LOGGER = logging.getLogger("generate_oracle_variants")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate oracle segmentation variants from MFA TextGrids.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--manifest", required=True, help="Input split manifest JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional utterance limit.")
    return parser.parse_args()


def build_variant_record(
    generated,
    source_row,
    output_audio_path,
):
    record = generated.to_dict()
    record.update(
        {
            "variant_id": generated.variant_id,
            "source_utterance_id": source_row["utterance_id"],
            "split": source_row["split"],
            "source_audio_path": source_row["audio_path"],
            "generated_audio_path": str(output_audio_path.resolve()),
            "textgrid_path": source_row["textgrid_path"],
            "reference": source_row["transcript"],
        }
    )
    return record


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(args.logging_config)
    LOGGER.info("START generate_oracle_variants | manifest=%s limit=%s", args.manifest, args.limit)

    source_manifest_path = Path(args.manifest).resolve()
    records = read_jsonl(source_manifest_path)
    limit = args.limit
    if limit is not None:
        records = records[:limit]

    output_root = ensure_dir(config["oracle"]["audio_output_path"])
    variant_manifest_dir = ensure_dir(config["manifest"]["output_path"])
    min_variant_duration_sec = float(config["oracle"]["min_variant_duration_sec"])

    variant_records: list[dict] = []
    experiment_counter: Counter[str] = Counter()
    skipped_counter: Counter[str] = Counter()

    for row in records:
        utterance_id = row["utterance_id"]
        split = row["split"]
        split_output_root = ensure_dir(output_root / split)

        try:
            audio, sample_rate = read_audio(row["audio_path"])
            alignment = parse_textgrid(row["textgrid_path"])
        except Exception:
            LOGGER.exception("Failed to prepare source utterance: %s", utterance_id)
            skipped_counter["source_load_error"] += 1
            continue

        leading_dir = ensure_dir(split_output_root / "leading")
        trailing_dir = ensure_dir(split_output_root / "trailing")
        pause_dir = ensure_dir(split_output_root / "pause_cut")

        for shift_ms in config["oracle"]["leading_ms"]:
            condition, variant_audio = generate_leading_variant(audio, sample_rate, alignment, shift_ms)
            duration_sec = len(variant_audio) / sample_rate
            if duration_sec < min_variant_duration_sec:
                skipped_counter["leading_too_short"] += 1
                continue
            variant_id = f"{utterance_id}__{condition}"
            output_path = leading_dir / f"{variant_id}.wav"
            write_audio(output_path, variant_audio, sample_rate)
            generated = GeneratedVariant(
                variant_id=variant_id,
                source_utterance_id=utterance_id,
                split=split,
                experiment_type="leading",
                condition=condition,
                parameter_value_ms=shift_ms,
                pause_bucket=None,
                pause_cut_ratio=None,
                pause_index=None,
                generated_duration_sec=duration_sec,
                sample_rate=sample_rate,
                transcript_reference=row["transcript"],
                metadata={},
            )
            variant_records.append(build_variant_record(generated, row, output_path))
            experiment_counter["leading"] += 1

        for shift_ms in config["oracle"]["trailing_ms"]:
            condition, variant_audio = generate_trailing_variant(audio, sample_rate, alignment, shift_ms)
            duration_sec = len(variant_audio) / sample_rate
            if duration_sec < min_variant_duration_sec:
                skipped_counter["trailing_too_short"] += 1
                continue
            variant_id = f"{utterance_id}__{condition}"
            output_path = trailing_dir / f"{variant_id}.wav"
            write_audio(output_path, variant_audio, sample_rate)
            generated = GeneratedVariant(
                variant_id=variant_id,
                source_utterance_id=utterance_id,
                split=split,
                experiment_type="trailing",
                condition=condition,
                parameter_value_ms=shift_ms,
                pause_bucket=None,
                pause_cut_ratio=None,
                pause_index=None,
                generated_duration_sec=duration_sec,
                sample_rate=sample_rate,
                transcript_reference=row["transcript"],
                metadata={},
            )
            variant_records.append(build_variant_record(generated, row, output_path))
            experiment_counter["trailing"] += 1

        pause_variants = generate_pause_variants(
            audio=audio,
            sample_rate=sample_rate,
            alignment=alignment,
            pause_buckets_ms=config["oracle"]["pause_buckets_ms"],
            pause_cut_positions=config["oracle"]["pause_cut_positions"],
            pause_min_duration_ms=config["oracle"]["pause_min_duration_ms"],
        )

        for generated, variant_audio in pause_variants:
            if generated.generated_duration_sec < min_variant_duration_sec:
                skipped_counter["pause_too_short"] += 1
                continue
            variant_id = f"{utterance_id}__{generated.condition}_i{generated.pause_index:02d}"
            output_path = pause_dir / f"{variant_id}.wav"
            write_audio(output_path, variant_audio, sample_rate)
            generated.variant_id = variant_id
            generated.source_utterance_id = utterance_id
            generated.split = split
            generated.transcript_reference = row["transcript"]
            variant_records.append(build_variant_record(generated, row, output_path))
            experiment_counter["pause_cut"] += 1

    output_manifest_path = variant_manifest_dir / f"{source_manifest_path.stem}_oracle_variants.jsonl"
    write_jsonl(output_manifest_path, variant_records)
    LOGGER.info("Variant manifest written to %s", output_manifest_path)
    LOGGER.info("Generated counts: %s", dict(experiment_counter))
    LOGGER.info("Skipped counts: %s", dict(skipped_counter))
    LOGGER.info("END generate_oracle_variants | manifest=%s variants=%d", source_manifest_path, len(variant_records))


if __name__ == "__main__":
    main()
