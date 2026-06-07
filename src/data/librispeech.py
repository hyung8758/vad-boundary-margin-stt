import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from src.audio.io import get_audio_duration


LOGGER = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    utterance_id: str
    audio_path: str
    transcript: str
    split: str
    speaker_id: str
    chapter_id: str
    textgrid_path: str
    duration_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def parse_utterance_id(utterance_id):
    parts = utterance_id.split("-")
    speaker_id = parts[0]
    chapter_id = parts[1]
    return speaker_id, chapter_id


def load_transcripts_for_split(split_root):
    transcripts = {}
    for transcript_path in sorted(Path(split_root).rglob("*.trans.txt")):
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    utterance_id, transcript = line.split(" ", 1)
                except ValueError:
                    LOGGER.warning("Malformed transcript line in %s: %s", transcript_path, line)
                    continue
                transcripts[utterance_id] = transcript.strip()
    return transcripts


def textgrid_path_for_utterance(textgrid_path, split, utterance_id):
    return Path(textgrid_path) / split / f"{utterance_id}.TextGrid"


def build_manifest_entries(
    data_path,
    textgrid_path,
    split,
    skip_missing_textgrid=True,
    limit=None,
):
    split_root = Path(data_path) / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    transcripts = load_transcripts_for_split(split_root)
    audio_paths = sorted(split_root.rglob("*.flac"))

    entries = []
    stats = {
        "audio_found": len(audio_paths),
        "missing_transcript": 0,
        "missing_textgrid": 0,
        "invalid_audio": 0,
        "valid": 0,
    }

    for audio_path in audio_paths:
        utterance_id = audio_path.stem
        if utterance_id not in transcripts:
            stats["missing_transcript"] += 1
            continue
        transcript = transcripts[utterance_id]

        utterance_textgrid_path = textgrid_path_for_utterance(textgrid_path, split, utterance_id)
        if not utterance_textgrid_path.exists():
            stats["missing_textgrid"] += 1
            if skip_missing_textgrid:
                continue

        try:
            duration_sec = get_audio_duration(audio_path)
        except Exception:
            stats["invalid_audio"] += 1
            LOGGER.exception("Failed to read audio metadata: %s", audio_path)
            continue

        speaker_id, chapter_id = parse_utterance_id(utterance_id)
        entries.append(
            ManifestEntry(
                utterance_id=utterance_id,
                audio_path=str(audio_path.resolve()),
                transcript=transcript,
                split=split,
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                textgrid_path=str(utterance_textgrid_path.resolve()),
                duration_sec=duration_sec,
            )
        )
        stats["valid"] += 1
        if limit is not None and len(entries) >= limit:
            break

    return entries, stats
