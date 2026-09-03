from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from transformers import TrainerCallback

_WER = evaluate.load("wer")
_CER = evaluate.load("cer")


def build_compute_metrics(processor):
    pad_id = processor.tokenizer.pad_token_id

    def _compute(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = pad_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = _WER.compute(predictions=pred_str, references=label_str)
        cer = _CER.compute(predictions=pred_str, references=label_str)

        accs, all_pred, all_lbl = [], [], []
        for i in range(len(pred_ids)):
            real_len = (pred.label_ids[i] != pad_id).sum()
            if real_len > 0:
                sp = pred_ids[i][:real_len]
                sl = pred.label_ids[i][:real_len]
                accs.append((sp == sl).mean())
                all_pred.extend(sp.tolist())
                all_lbl.extend(sl.tolist())

        accuracy = float(np.mean(accs)) if accs else 0.0
        precision = 0.0
        if all_pred and all_lbl:
            precision = precision_score(
                all_lbl, all_pred,
                average="macro", zero_division=0,
                labels=np.unique(all_lbl + all_pred),
            )
        return {"wer": wer, "cer": cer, "accuracy": accuracy, "precision": precision}

    return _compute


@dataclass
class MetricsTracker:
    train_metrics: List[dict] = field(default_factory=list)
    eval_metrics: List[dict] = field(default_factory=list)

    def add_eval(self, step, loss, wer, cer, accuracy, precision):
        self.eval_metrics.append({
            "step": step, "eval_loss": loss, "eval_wer": wer,
            "eval_cer": cer, "eval_accuracy": accuracy, "eval_precision": precision,
        })

    def best(self) -> Optional[dict]:
        if not self.eval_metrics:
            return None
        wers = [m["eval_wer"] for m in self.eval_metrics if m["eval_wer"] is not None]
        if not wers:
            return None
        idx = int(np.argmin(wers))
        return self.eval_metrics[idx]

    def save_csv(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rows = self.train_metrics + self.eval_metrics
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)


class MetricsCallback(TrainerCallback):
    def __init__(self, tracker: MetricsTracker):
        super().__init__()
        self.tracker = tracker

    def on_evaluate(self, args, state, control, **kwargs):
        m = kwargs.get("metrics", {}) or {}
        self.tracker.add_eval(
            step=state.global_step,
            loss=m.get("eval_loss"),
            wer=m.get("eval_wer"),
            cer=m.get("eval_cer"),
            accuracy=m.get("eval_accuracy"),
            precision=m.get("eval_precision"),
        )
