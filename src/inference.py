from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .preprocessing import reduce_noise


def load_pretrained(model_dir: Union[str, Path], processor_dir: Union[str, Path]):
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model.eval()
    return model, processor


def load_audio(path: Union[str, Path], target_sr: int = 16_000) -> np.ndarray:
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio


def transcribe(
    audio: np.ndarray,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    denoise: bool = True,
) -> str:
    sr = processor.feature_extractor.sampling_rate
    if denoise:
        audio = reduce_noise(audio, sr)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    device = next(model.parameters()).device
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def transcribe_file(
    path: Union[str, Path],
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    denoise: bool = True,
) -> str:
    audio = load_audio(path, processor.feature_extractor.sampling_rate)
    return transcribe(audio, model, processor, denoise=denoise)
