from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    dataset_name: str = "IndabaxSenegal/asr-wolof-dataset"
    dataset_test_name: str = "IndabaxSenegal/asr-wolof-dataset-test"
    output_dir: Path = Path("./wav2vec2-wolof-results")

    sampling_rate: int = 16_000
    max_input_length: int = 18
    min_input_length: int = 1
    max_tokens: int = 310

    batch_size_train: int = 8
    batch_size_eval: int = 8
    num_epochs: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 5e-3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1

    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    seed: int = 42
    dataloader_num_workers: int = 4


CONFIG = Config()
