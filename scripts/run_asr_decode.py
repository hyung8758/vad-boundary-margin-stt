import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import ensure_dir, load_config, read_jsonl, set_cuda_visible_devices, setup_logging, write_jsonl


LOGGER = logging.getLogger("run_asr_decode")


def parse_args():
    parser = argparse.ArgumentParser(description="Decode manifest entries with wav2vec2-base-960h.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--logging-config", default=str(PROJECT_ROOT / "conf" / "logging.yaml"))
    parser.add_argument("--manifest", required=True, help="Baseline or variant manifest JSONL.")
    parser.add_argument("--exp-subdir", default="", help="Optional experiment subdirectory under exp_path, e.g. step1 or step2.")
    parser.add_argument("--output", default=None, help="Optional decode JSONL output path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional decode limit.")
    return parser.parse_args()


def build_decode_record(row, decode_result):
    if "generated_audio_path" in row:
        variant_id = row["variant_id"]
        source_utterance_id = row["source_utterance_id"]
        reference = row["reference"]
        experiment_type = row["experiment_type"]
        condition = row["condition"]
        audio_path = row["generated_audio_path"]
    else:
        variant_id = row["utterance_id"]
        source_utterance_id = row["utterance_id"]
        reference = row["transcript"]
        experiment_type = "baseline"
        condition = "baseline"
        audio_path = row["audio_path"]

    return {
        "variant_id": variant_id,
        "source_utterance_id": source_utterance_id,
        "experiment_type": experiment_type,
        "condition": condition,
        "split": row["split"],
        "hypothesis": decode_result["hypothesis"],
        "reference": reference,
        "audio_path": audio_path,
        "asr_language": decode_result["language"],
        "asr_language_probability": decode_result["language_probability"],
        "decoded_duration_sec": decode_result["duration_sec"],
        "segment_count": decode_result["segment_count"],
    }


def get_audio_path(row):
    if "generated_audio_path" in row:
        return row["generated_audio_path"]
    return row["audio_path"]


def decode_one_row(row, runner, config):
    audio_path = get_audio_path(row)
    decode_result = runner.transcribe(audio_path)
    record = build_decode_record(row, decode_result)
    record["model_name"] = config["asr"]["model_name"]
    record["device"] = config["asr"]["device"]
    record["device_index"] = runner.settings.device_index
    record["physical_device_index"] = runner.settings.physical_device_index
    return record


def decode_rows(rows, runner, config):
    return [decode_one_row(row, runner, config) for row in rows]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_cuda_visible_devices(config)
    from src.asr import create_asr_runners, get_asr_model_tag

    setup_logging(args.logging_config)
    LOGGER.info("START run_asr_decode | manifest=%s exp_subdir=%s limit=%s output=%s", args.manifest, args.exp_subdir, args.limit, args.output)

    input_manifest = Path(args.manifest).resolve()
    rows = read_jsonl(input_manifest)
    limit = args.limit
    if limit is not None:
        rows = rows[:limit]

    exp_root = Path(config["exp_path"])
    if args.exp_subdir:
        exp_root = exp_root / args.exp_subdir
    output_dir = ensure_dir(exp_root / "decodes")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else output_dir / f"{input_manifest.stem}__{get_asr_model_tag(config)}.jsonl"
    )

    runners = create_asr_runners(config)
    results = []
    num_workers = int(config["asr"]["num_workers"])

    if num_workers > 1 and len(runners) > 1:
        worker_count = min(num_workers, len(runners))
        row_groups = []
        for group_index in range(worker_count):
            row_groups.append(rows[group_index::worker_count])

        futures = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for worker_index in range(worker_count):
                group = row_groups[worker_index]
                runner = runners[worker_index]
                LOGGER.info("Worker %d uses GPU %s for %d files", worker_index, runner.settings.device_index, len(group))
                futures.append(executor.submit(decode_rows, group, runner, config))

            for future in futures:
                results.extend(future.result())
    else:
        runner = runners[0]
        for index, row in enumerate(rows, start=1):
            audio_path = get_audio_path(row)
            LOGGER.info("Decoding %d/%d: %s", index, len(rows), audio_path)
            results.append(decode_one_row(row, runner, config))

    write_jsonl(output_path, results)
    LOGGER.info("Decode output written to %s", output_path)
    LOGGER.info("END run_asr_decode | manifest=%s exp_subdir=%s rows=%d output=%s", input_manifest, args.exp_subdir, len(results), output_path)


if __name__ == "__main__":
    main()
