from pathlib import Path

import numpy as np
import torch
import torchaudio


def read_audio(path):
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    return audio, sample_rate


def write_audio(path, audio, sample_rate):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(str(output_path), waveform, sample_rate)


def get_audio_duration(path):
    waveform, sample_rate = torchaudio.load(str(path))
    return float(waveform.shape[-1]) / float(sample_rate)


def seconds_to_samples(time_sec, sample_rate):
    return max(0, int(round(time_sec * sample_rate)))


def clamp_time(time_sec, minimum, maximum):
    return max(minimum, min(time_sec, maximum))


def slice_audio(audio, sample_rate, start_sec, end_sec):
    start_index = seconds_to_samples(start_sec, sample_rate)
    end_index = seconds_to_samples(end_sec, sample_rate)
    if end_index <= start_index:
        return np.array([], dtype=np.float32)
    return audio[start_index:end_index].copy()


def concatenate_segments(segments):
    valid_segments = [segment for segment in segments if segment.size > 0]
    if not valid_segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(valid_segments).astype(np.float32)
