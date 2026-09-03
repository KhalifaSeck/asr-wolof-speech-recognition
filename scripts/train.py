from pathlib import Path

from src.collator import DataCollatorCTCWithPadding
from src.config import CONFIG
from src.data import clean_text, load_wolof_datasets
from src.metrics import MetricsTracker
from src.model import build_model
from src.preprocessing import build_length_filter, prepare_test_sample, prepare_train_sample
from src.seeding import set_seed
from src.trainer_setup import build_trainer
from src.vocab import build_processor, build_vocab, save_vocab


def main() -> None:
    set_seed(CONFIG.seed)

    print("[1/6] Loading datasets")
    dataset, dataset_test = load_wolof_datasets()
    dataset = clean_text(dataset)

    print("[2/6] Building vocabulary and processor")
    vocab = build_vocab(dataset["train"], dataset["test"])
    save_vocab(vocab, Path(CONFIG.output_dir) / "vocab.json")
    processor = build_processor(vocab, CONFIG.sampling_rate)
    processor.save_pretrained(Path(CONFIG.output_dir) / "processor")

    print("[3/6] Encoding datasets")
    encode_train = prepare_train_sample(processor)
    encode_test = prepare_test_sample(processor)

    dataset_encoded = {
        "train": dataset["train"].map(
            encode_train, remove_columns=dataset["train"].column_names, num_proc=CONFIG.dataloader_num_workers
        ),
        "test": dataset["test"].map(
            encode_train, remove_columns=dataset["test"].column_names, num_proc=CONFIG.dataloader_num_workers
        ),
    }
    dataset_test_encoded = dataset_test.map(
        encode_test, remove_columns=dataset_test.column_names, num_proc=CONFIG.dataloader_num_workers
    )

    train_filter = build_length_filter(
        CONFIG.min_input_length, CONFIG.max_input_length, CONFIG.sampling_rate, CONFIG.max_tokens, require_labels=True
    )
    eval_filter = build_length_filter(
        CONFIG.min_input_length, CONFIG.max_input_length, CONFIG.sampling_rate, CONFIG.max_tokens, require_labels=False
    )
    dataset_encoded["train"] = dataset_encoded["train"].filter(train_filter)
    dataset_encoded["test"] = dataset_encoded["test"].filter(eval_filter)
    dataset_test_encoded = dataset_test_encoded.filter(eval_filter)

    print(f"    Train: {len(dataset_encoded['train'])} | Val: {len(dataset_encoded['test'])} | Test: {len(dataset_test_encoded)}")

    print("[4/6] Building model")
    model = build_model(CONFIG.model_name, processor)

    print("[5/6] Training")
    collator = DataCollatorCTCWithPadding(processor=processor)
    tracker = MetricsTracker()
    trainer = build_trainer(model, processor, dataset_encoded, collator, tracker)
    trainer.train()

    print("[6/6] Saving model and metrics")
    trainer.model.save_pretrained(Path(CONFIG.output_dir) / "model")
    processor.save_pretrained(Path(CONFIG.output_dir) / "processor")
    tracker.save_csv(Path(CONFIG.output_dir) / "training_metrics.csv")

    best = tracker.best()
    if best:
        print(f"Best WER: {best['eval_wer']:.4f} at step {best['step']}")


if __name__ == "__main__":
    main()
