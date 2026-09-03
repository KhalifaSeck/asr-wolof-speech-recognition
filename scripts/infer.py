import argparse
from pathlib import Path

from src.config import CONFIG
from src.inference import load_pretrained, transcribe_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path, help="Path to an audio file (wav, mp3, flac)")
    p.add_argument("--model-dir", type=Path, default=Path(CONFIG.output_dir) / "model")
    p.add_argument("--processor-dir", type=Path, default=Path(CONFIG.output_dir) / "processor")
    p.add_argument("--no-denoise", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, processor = load_pretrained(args.model_dir, args.processor_dir)
    text = transcribe_file(args.audio, model, processor, denoise=not args.no_denoise)
    print(text)


if __name__ == "__main__":
    main()
