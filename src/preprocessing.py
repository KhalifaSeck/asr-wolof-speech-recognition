import warnings

import noisereduce as nr
import numpy as np


def reduce_noise(audio: np.ndarray, sampling_rate: int) -> np.ndarray:
    if len(audio) == 0:
        return audio
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nr.reduce_noise(
                y=audio,
                sr=sampling_rate,
                stationary=True,
                prop_decrease=0.4,
                n_std_thresh_stationary=2.5,
                n_fft=min(512, len(audio) // 4),
            )
    except Exception:
        return audio


def prepare_train_sample(processor):
    sr = processor.feature_extractor.sampling_rate

    def _fn(batch):
        audio = batch["audio"]["array"]
        if audio is None or len(audio) == 0:
            return None
        cleaned = reduce_noise(audio, sr)
        batch["input_values"] = processor(cleaned, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    return _fn


def prepare_test_sample(processor):
    sr = processor.feature_extractor.sampling_rate

    def _fn(batch):
        audio = batch["audio"]["array"]
        if audio is None or len(audio) == 0:
            return None
        cleaned = reduce_noise(audio, sr)
        batch["input_values"] = processor(cleaned, sampling_rate=sr).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        return batch

    return _fn


def build_length_filter(min_input_length: int, max_input_length: int, sampling_rate: int, max_tokens: int, require_labels: bool = True):
    min_len = min_input_length * sampling_rate
    max_len = max_input_length * sampling_rate

    def _filter(sample):
        length_ok = min_len < sample["input_length"] < max_len
        if not require_labels:
            return length_ok
        return length_ok and len(sample["labels"]) >= 4 and len(sample["labels"]) <= max_tokens

    return _filter
