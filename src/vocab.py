import json
import tempfile
from pathlib import Path

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

BASIC_LATIN = "abcdefghijklmnopqrstuvwxyz"
WOLOF_SPECIAL = "àáâãäçèéêëîïñòóôõùûāńŋ"
EXCLUDED_CHARS = {
    "\n", "\r", "\t",
    "$", "&", "(", ")", "*", "/", "=", "^", "_", "~", "£", "μ", " ̈",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


def _extract_all_chars(dataset):
    def _map(batch):
        all_text = " ".join(batch["text"])
        return {"vocab": [list(set(all_text))], "all_text": [all_text]}

    return dataset.map(
        _map,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )


def _is_valid_char(char: str) -> bool:
    if char in EXCLUDED_CHARS:
        return False
    lower = char.lower()
    return lower in BASIC_LATIN or char == " " or lower in WOLOF_SPECIAL


def build_vocab(train_dataset, test_dataset) -> dict:
    vocab_train = _extract_all_chars(train_dataset)
    vocab_test = _extract_all_chars(test_dataset)

    all_chars = set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0])
    filtered = [c for c in all_chars if _is_valid_char(c)]

    vocab = {char: idx for idx, char in enumerate(sorted(filtered))}
    if " " in vocab:
        vocab["|"] = vocab[" "]
        del vocab[" "]
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def save_vocab(vocab: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def build_processor(vocab: dict, sampling_rate: int) -> Wav2Vec2Processor:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as fh:
        json.dump(vocab, fh)
        vocab_path = fh.name

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    Path(vocab_path).unlink(missing_ok=True)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
