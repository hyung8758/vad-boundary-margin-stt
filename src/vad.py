from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from src.audio.io import concatenate_segments


@dataclass
class SileroVadSettings:
    threshold: float
    min_silence_duration_ms: int
    speech_pad_ms: int
    max_speech_duration_s: float
    target_sample_rate: int


def resample_audio(audio, source_rate, target_rate):
    if source_rate == target_rate:
        return audio.astype(np.float32)

    source_positions = np.arange(len(audio), dtype=np.float32)
    target_length = int(round(len(audio) * (target_rate / source_rate)))
    if target_length <= 1:
        return np.array([], dtype=np.float32)

    target_positions = np.linspace(0, len(audio) - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


class SileroVadRunner:
    def __init__(self, settings):
        self.settings = settings
        self.model = load_silero_vad()

    def detect_segments(self, audio, sample_rate):
        vad_audio = resample_audio(audio, sample_rate, self.settings.target_sample_rate)
        vad_tensor = torch.from_numpy(vad_audio)
        segments = get_speech_timestamps(
            vad_tensor,
            self.model,
            sampling_rate=self.settings.target_sample_rate,
            threshold=self.settings.threshold,
            min_silence_duration_ms=self.settings.min_silence_duration_ms,
            speech_pad_ms=self.settings.speech_pad_ms,
            max_speech_duration_s=self.settings.max_speech_duration_s,
        )
        return vad_audio, self.settings.target_sample_rate, segments

    def reconstruct_audio(self, audio, segments):
        pieces = []
        for segment in segments:
            pieces.append(audio[segment["start"] : segment["end"]])
        return concatenate_segments(pieces)


def create_silero_vad_runner(config):
    settings = SileroVadSettings(
        threshold=float(config["silero_vad"]["threshold"]),
        min_silence_duration_ms=int(config["silero_vad"]["min_silence_duration_ms"]),
        speech_pad_ms=int(config["silero_vad"]["speech_pad_ms"]),
        max_speech_duration_s=float(config["silero_vad"]["max_speech_duration_s"]),
        target_sample_rate=int(config["silero_vad"]["target_sample_rate"]),
    )
    return SileroVadRunner(settings)
