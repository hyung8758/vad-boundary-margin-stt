from dataclasses import asdict, dataclass
from pathlib import Path

from textgrid import TextGrid


@dataclass
class WordInterval:
    text: str
    start: float
    end: float


@dataclass
class PauseInterval:
    previous_word: str
    next_word: str
    start: float
    end: float
    duration_sec: float
    index: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UtteranceAlignment:
    textgrid_path: str
    utterance_start: float
    utterance_end: float
    words: list


def get_words_tier(grid):
    for tier in grid.tiers:
        tier_name = tier.name.strip().lower()
        if tier_name == "words":
            return tier
    raise ValueError("TextGrid does not contain a 'words' tier")


def parse_textgrid(path):
    grid = TextGrid.fromFile(str(path))
    word_tier = get_words_tier(grid)

    words = []
    for interval in word_tier.intervals:
        text = (interval.mark or "").strip()
        if not text:
            continue
        start = float(interval.minTime)
        end = float(interval.maxTime)
        if end <= start:
            continue
        words.append(WordInterval(text=text, start=start, end=end))

    if words:
        utterance_start = words[0].start
        utterance_end = words[-1].end
    else:
        utterance_start = float(word_tier.minTime)
        utterance_end = float(word_tier.maxTime)

    return UtteranceAlignment(
        textgrid_path=str(Path(path).resolve()),
        utterance_start=utterance_start,
        utterance_end=utterance_end,
        words=words,
    )


def compute_inter_word_pauses(
    alignment,
    min_duration_ms=0,
):
    pauses = []
    threshold_sec = min_duration_ms / 1000.0
    for index in range(len(alignment.words) - 1):
        current_word = alignment.words[index]
        next_word = alignment.words[index + 1]
        pause_start = current_word.end
        pause_end = next_word.start
        pause_duration = max(0.0, pause_end - pause_start)
        if pause_duration < threshold_sec:
            continue
        pauses.append(
            PauseInterval(
                previous_word=current_word.text,
                next_word=next_word.text,
                start=pause_start,
                end=pause_end,
                duration_sec=pause_duration,
                index=index,
            )
        )
    return pauses
