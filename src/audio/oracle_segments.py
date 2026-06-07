from dataclasses import asdict, dataclass

from src.audio.io import clamp_time, concatenate_segments, slice_audio
from src.data.textgrid_parser import compute_inter_word_pauses


@dataclass
class GeneratedVariant:
    variant_id: str
    source_utterance_id: str
    split: str
    experiment_type: str
    condition: str
    parameter_value_ms: int
    pause_bucket: str
    pause_cut_ratio: float
    pause_index: int
    generated_duration_sec: float
    sample_rate: int
    transcript_reference: str
    metadata: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        metadata = data.pop("metadata")
        data.update(metadata)
        return data


def format_signed_ms(prefix, value_ms):
    sign = "p" if value_ms >= 0 else "m"
    return f"{prefix}_{sign}{abs(value_ms):03d}"


def pause_bucket_label(bucket_start_ms, bucket_end_ms):
    upper = "inf" if bucket_end_ms is None else str(bucket_end_ms)
    return f"b{bucket_start_ms}_{upper}"


def pause_ratio_label(ratio):
    return f"r{int(round(ratio * 100)):02d}"


def generate_leading_variant(
    audio,
    sample_rate,
    alignment,
    shift_ms,
):
    duration_sec = len(audio) / sample_rate
    start_sec = clamp_time(alignment.utterance_start - (shift_ms / 1000.0), 0.0, duration_sec)
    end_sec = clamp_time(alignment.utterance_end, 0.0, duration_sec)
    condition = format_signed_ms("leading", shift_ms)
    return condition, slice_audio(audio, sample_rate, start_sec, end_sec)


def generate_trailing_variant(
    audio,
    sample_rate,
    alignment,
    shift_ms,
):
    duration_sec = len(audio) / sample_rate
    start_sec = clamp_time(alignment.utterance_start, 0.0, duration_sec)
    end_sec = clamp_time(alignment.utterance_end + (shift_ms / 1000.0), 0.0, duration_sec)
    condition = format_signed_ms("trailing", shift_ms)
    return condition, slice_audio(audio, sample_rate, start_sec, end_sec)


def pause_matches_bucket(pause, bucket_start_ms, bucket_end_ms):
    duration_ms = pause.duration_sec * 1000.0
    if duration_ms < bucket_start_ms:
        return False
    if bucket_end_ms is None:
        return True
    return duration_ms < bucket_end_ms


def generate_pause_variants(
    audio,
    sample_rate,
    alignment,
    pause_buckets_ms,
    pause_cut_positions,
    pause_min_duration_ms,
):
    variants = []
    pauses = compute_inter_word_pauses(alignment, min_duration_ms=pause_min_duration_ms)

    for pause in pauses:
        for bucket_start_ms, bucket_end_ms in pause_buckets_ms:
            if not pause_matches_bucket(pause, bucket_start_ms, bucket_end_ms):
                continue

            bucket_label = pause_bucket_label(bucket_start_ms, bucket_end_ms)
            for ratio in pause_cut_positions:
                cut_time = pause.start + (ratio * pause.duration_sec)
                left_audio = slice_audio(audio, sample_rate, alignment.utterance_start, cut_time)
                right_audio = slice_audio(audio, sample_rate, pause.end, alignment.utterance_end)
                variant_audio = concatenate_segments([left_audio, right_audio])
                condition = f"pause_{bucket_label}_{pause_ratio_label(ratio)}"
                generated = GeneratedVariant(
                    variant_id="",
                    source_utterance_id="",
                    split="",
                    experiment_type="pause_cut",
                    condition=condition,
                    parameter_value_ms=None,
                    pause_bucket=bucket_label,
                    pause_cut_ratio=ratio,
                    pause_index=pause.index,
                    generated_duration_sec=len(variant_audio) / sample_rate,
                    sample_rate=sample_rate,
                    transcript_reference="",
                    metadata={
                        "pause_start_sec": pause.start,
                        "pause_end_sec": pause.end,
                        "pause_duration_ms": round(pause.duration_sec * 1000.0, 3),
                        "pause_previous_word": pause.previous_word,
                        "pause_next_word": pause.next_word,
                        "pause_cut_time_sec": cut_time,
                        "pause_retained_ms": round((cut_time - pause.start) * 1000.0, 3),
                    },
                )
                variants.append((generated, variant_audio))

    return variants
