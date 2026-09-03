from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch

from .config import CONFIG

_WER = evaluate.load("wer")
_CER = evaluate.load("cer")


def evaluate_model(model, processor, encoded_dataset, batch_size: int = 4) -> dict:
    model.eval()
    device = next(model.parameters()).device
    pad_id = processor.tokenizer.pad_token_id
    predictions, references = [], []

    with torch.no_grad():
        for start in range(0, len(encoded_dataset), batch_size):
            batch = encoded_dataset[start:start + batch_size]
            input_values = [torch.tensor(x) for x in batch["input_values"]]
            padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True).to(device)

            logits = model(padded).logits
            pred_ids = torch.argmax(logits, dim=-1)
            predictions.extend(processor.batch_decode(pred_ids))

            if "labels" in batch:
                labels = [np.array(l) for l in batch["labels"]]
                for lbl in labels:
                    lbl[lbl == -100] = pad_id
                    references.append(processor.decode(lbl, group_tokens=False))

    result = {"num_samples": len(predictions), "predictions": predictions}
    if references:
        result["wer"] = _WER.compute(predictions=predictions, references=references)
        result["cer"] = _CER.compute(predictions=predictions, references=references)
        result["references"] = references
    return result


def save_predictions(result: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"prediction": p} for p in result["predictions"]]
    if "references" in result:
        for i, r in enumerate(result["references"]):
            rows[i]["reference"] = r
    pd.DataFrame(rows).to_csv(path, index=False)
