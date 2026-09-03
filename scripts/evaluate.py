import argparse
from pathlib import Path

from src.collator import DataCollatorCTCWithPadding
from src.config import CONFIG
from src.data import load_wolof_datasets
from src.evaluation import evaluate_model, save_predictions
from src.inference import load_pretrained
from src.preprocessing import build_length_filter, prepare_test_sample


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path(CONFIG.output_dir) / "model")
    p.add_argument("--processor-dir", type=Path, default=Path(CONFIG.output_dir) / "processor")
    p.add_argument("--output-csv", type=Path, default=Path("reports/test_predictions.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading model and processor")
    model, processor = load_pretrained(args.model_dir, args.processor_dir)

    print("Loading test dataset")
    _, dataset_test = load_wolof_datasets()
    encoder = prepare_test_sample(processor)
    encoded = dataset_test.map(encoder, remove_columns=dataset_test.column_names, num_proc=CONFIG.dataloader_num_workers)
    encoded = encoded.filter(
        build_length_filter(
            CONFIG.min_input_length, CONFIG.max_input_length, CONFIG.sampling_rate, CONFIG.max_tokens, require_labels=False
        )
    )

    print(f"Evaluating on {len(encoded)} samples")
    result = evaluate_model(model, processor, encoded)

    if "wer" in result:
        print(f"WER: {result['wer']:.4f}  CER: {result['cer']:.4f}")

    save_predictions(result, args.output_csv)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()
