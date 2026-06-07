import re
from dataclasses import dataclass

import torch
import torchaudio

from src.audio.io import read_audio


WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class Wav2Vec2Settings:
    model_name: str
    device: str
    device_index: int


def get_asr_model_tag(config):
    model_name = config["asr"]["model_name"]
    return model_name.replace("-", "_")


def get_device_indices(config):
    device_index = config["asr"]["device_index"]
    if isinstance(device_index, list):
        return [int(value) for value in device_index]
    return [int(device_index)]


def decode_tokens(token_ids, labels):
    blank_id = 0
    pieces = []
    previous_id = None

    for token_id in token_ids:
        if token_id == previous_id:
            continue
        previous_id = token_id
        if token_id == blank_id:
            continue
        token = labels[token_id]
        if token == "|":
            pieces.append(" ")
        else:
            pieces.append(token)

    text = "".join(pieces)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


class Wav2Vec2Runner:
    def __init__(self, settings):
        self.settings = settings
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.labels = self.bundle.get_labels()
        self.sample_rate = self.bundle.sample_rate

        if settings.device == "cuda":
            self.device = "cuda:%d" % settings.device_index
        else:
            self.device = "cpu"

        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()

    def transcribe(self, audio_path):
        audio, sample_rate = read_audio(audio_path)
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        waveform = waveform.to(self.device)

        with torch.inference_mode():
            emissions, _ = self.model(waveform)

        token_ids = torch.argmax(emissions, dim=-1)[0].detach().cpu().tolist()
        hypothesis = decode_tokens(token_ids, self.labels)

        return {
            "hypothesis": hypothesis,
            "language": "en",
            "language_probability": 1.0,
            "duration_sec": waveform.shape[-1] / sample_rate,
            "segment_count": 1,
        }


def create_asr_runners(config):
    if config["asr"]["model_name"] != "wav2vec2-base-960h":
        raise ValueError("Unsupported ASR model: %s" % config["asr"]["model_name"])

    if config["asr"]["device"] == "cpu":
        settings = Wav2Vec2Settings(
            model_name=config["asr"]["model_name"],
            device="cpu",
            device_index=0,
        )
        return [Wav2Vec2Runner(settings)]

    runners = []
    for device_index in get_device_indices(config):
        settings = Wav2Vec2Settings(
            model_name=config["asr"]["model_name"],
            device=config["asr"]["device"],
            device_index=device_index,
        )
        runners.append(Wav2Vec2Runner(settings))
    return runners
