import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_dataset_config, load_config
from src.data.librispeech import load_transcripts_for_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech utterance-level MFA corpus files and run Montreal Forced Aligner."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--dataset", default="librispeech", help="Dataset name under conf/base.yaml datasets.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to align. Defaults to all splits listed in conf/base.yaml.",
    )
    parser.add_argument("--dictionary", default="english_us_mfa", help="MFA dictionary name or path.")
    parser.add_argument("--acoustic-model", default="english_mfa", help="MFA acoustic model name or path.")
    parser.add_argument("--mfa-corpus-dir", default=None, help="Prepared MFA corpus root. Default: data/LibriSpeech/mfa_corpus")
    parser.add_argument("--textgrid-dir", default=None, help="TextGrid output root. Default: config datasets.<dataset>.textgrid_path")
    parser.add_argument("--jobs", type=int, default=4, help="Number of MFA jobs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional utterance limit per split.")
    parser.add_argument("--overwrite-corpus", action="store_true", help="Recreate prepared MFA corpus split directories.")
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio instead of creating symlinks.")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not pass MFA's --clean option. By default, MFA temporary files are reset.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare MFA corpus files; do not run alignment.")
    parser.add_argument("--dry-run", action="store_true", help="Print mfa commands without executing them.")
    return parser.parse_args()


def safe_unlink(path):
    if path.exists() or path.is_symlink():
        path.unlink()


def link_or_copy_audio(source_path, target_path, copy_audio):
    safe_unlink(target_path)
    if copy_audio:
        shutil.copy2(source_path, target_path)
    else:
        target_path.symlink_to(source_path.resolve())


def write_lab_file(path, transcript):
    path.write_text(transcript.strip() + "\n", encoding="utf-8")


def prepare_split_corpus(data_path, split, output_dir, limit=None, overwrite=False, copy_audio=False):
    split_root = Path(data_path) / split
    if not split_root.exists():
        raise FileNotFoundError(f"LibriSpeech split not found: {split_root}")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts_for_split(split_root)
    audio_paths = sorted(split_root.rglob("*.flac"))
    prepared = 0
    missing_transcript = 0

    for audio_path in audio_paths:
        utterance_id = audio_path.stem
        transcript = transcripts.get(utterance_id)
        if transcript is None:
            missing_transcript += 1
            continue

        target_audio_path = output_dir / audio_path.name
        target_lab_path = output_dir / f"{utterance_id}.lab"
        link_or_copy_audio(audio_path, target_audio_path, copy_audio)
        write_lab_file(target_lab_path, transcript)
        prepared += 1

        if limit is not None and prepared >= limit:
            break

    if prepared == 0:
        raise ValueError(f"No utterances prepared for split={split}")

    return {
        "audio_found": len(audio_paths),
        "prepared": prepared,
        "missing_transcript": missing_transcript,
    }


def build_mfa_command(args, corpus_dir, output_dir):
    command = [
        "mfa",
        "align",
        "--output_format",
        "long_textgrid",
        "-j",
        str(args.jobs),
    ]
    if not args.no_clean:
        command.append("--clean")
    command.extend(
        [
            str(corpus_dir),
            args.dictionary,
            args.acoustic_model,
            str(output_dir),
        ]
    )
    return command


def run_command(command, dry_run=False):
    print("[mfa]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    _, dataset = get_dataset_config(config, args.dataset)

    splits = args.splits or dataset["splits"]
    data_path = Path(dataset["data_path"])
    mfa_corpus_root = Path(args.mfa_corpus_dir).resolve() if args.mfa_corpus_dir else data_path / "mfa_corpus"
    textgrid_root = Path(args.textgrid_dir).resolve() if args.textgrid_dir else Path(dataset["textgrid_path"])

    for split in splits:
        split_corpus_dir = mfa_corpus_root / split
        split_textgrid_dir = textgrid_root / split

        stats = prepare_split_corpus(
            data_path=data_path,
            split=split,
            output_dir=split_corpus_dir,
            limit=args.limit,
            overwrite=args.overwrite_corpus,
            copy_audio=args.copy_audio,
        )
        print(
            "[prepared] split=%s corpus=%s prepared=%d audio_found=%d missing_transcript=%d"
            % (split, split_corpus_dir, stats["prepared"], stats["audio_found"], stats["missing_transcript"])
        )

        if args.prepare_only:
            continue

        split_textgrid_dir.mkdir(parents=True, exist_ok=True)
        command = build_mfa_command(args, split_corpus_dir, split_textgrid_dir)
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
