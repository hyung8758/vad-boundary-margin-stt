from dataclasses import dataclass

from src.common import get_split_group


@dataclass
class VadPolicySettings:
    policy_name: str
    leading_pad_ms: int
    trailing_pad_ms: int


def milliseconds_to_samples(value_ms, sample_rate):
    return max(0, int(round((value_ms / 1000.0) * sample_rate)))


def normalize_segments(segments, audio_length):
    normalized = []
    for segment in segments:
        start = max(0, int(segment["start"]))
        end = min(audio_length, int(segment["end"]))
        if end <= start:
            continue
        normalized.append({"start": start, "end": end})
    normalized.sort(key=lambda segment: segment["start"])
    return normalized


def apply_vad_policy(segments, sample_rate, audio_length, settings):
    base_segments = normalize_segments(segments, audio_length)
    if not base_segments:
        return []

    leading_pad_samples = milliseconds_to_samples(settings.leading_pad_ms, sample_rate)
    trailing_pad_samples = milliseconds_to_samples(settings.trailing_pad_ms, sample_rate)

    adjusted = []
    for index, segment in enumerate(base_segments):
        start = max(0, segment["start"] - leading_pad_samples)
        end = min(audio_length, segment["end"] + trailing_pad_samples)

        if index < len(base_segments) - 1:
            next_segment = base_segments[index + 1]
            next_start_with_leading = max(0, next_segment["start"] - leading_pad_samples)
            if end > next_start_with_leading:
                end = next_start_with_leading

        if end <= start:
            continue
        adjusted.append(
            {
                "start": start,
                "end": end,
            }
        )

    return normalize_segments(adjusted, audio_length)


def create_vad_policy_settings(config):
    section = config["vad_policy"]
    return {
        "policy_name": section["policy_name"],
        "clean": VadPolicySettings(
            policy_name=section["policy_name"],
            leading_pad_ms=int(section["clean"]["leading_pad_ms"]),
            trailing_pad_ms=int(section["clean"]["trailing_pad_ms"]),
        ),
        "other": VadPolicySettings(
            policy_name=section["policy_name"],
            leading_pad_ms=int(section["other"]["leading_pad_ms"]),
            trailing_pad_ms=int(section["other"]["trailing_pad_ms"]),
        ),
    }


def get_vad_policy_settings(policy_config, split_name):
    group = get_split_group(split_name)
    return policy_config[group]
