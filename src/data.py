import re

from datasets import Audio, load_dataset

from .config import CONFIG

CHARS_TO_IGNORE = re.compile(r"[\,\?\.\!\-\;\:\"\"\%\'\"�]")


def load_wolof_datasets(seed: int = CONFIG.seed):
    train_full = load_dataset(CONFIG.dataset_name, split="train")
    test_holdout = load_dataset(CONFIG.dataset_test_name, split="test")

    dataset = train_full.train_test_split(test_size=0.1, seed=seed)

    dataset = dataset.remove_columns(["duration", "file_name", "path"])
    test_holdout = test_holdout.remove_columns(["duration", "file_name", "path"])

    dataset = dataset.cast_column("audio", Audio(sampling_rate=CONFIG.sampling_rate))
    test_holdout = test_holdout.cast_column(
        "audio", Audio(sampling_rate=CONFIG.sampling_rate)
    )

    return dataset, test_holdout


def remove_special_characters(batch):
    batch["text"] = CHARS_TO_IGNORE.sub("", batch["text"]).lower() + " "
    return batch


def clean_text(dataset):
    return dataset.map(remove_special_characters)
