from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .config import CONFIG
from .metrics import MetricsCallback, MetricsTracker, build_compute_metrics


def build_training_args(steps_per_epoch: int):
    total_steps = steps_per_epoch * CONFIG.num_epochs
    warmup_steps = int(CONFIG.warmup_ratio * total_steps)

    return TrainingArguments(
        output_dir=str(CONFIG.output_dir),
        group_by_length=True,
        per_device_train_batch_size=CONFIG.batch_size_train,
        per_device_eval_batch_size=CONFIG.batch_size_eval,
        eval_strategy="steps",
        num_train_epochs=CONFIG.num_epochs,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=CONFIG.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=CONFIG.weight_decay,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_dir=f"{CONFIG.output_dir}/logs",
        seed=CONFIG.seed,
        dataloader_num_workers=CONFIG.dataloader_num_workers,
        report_to=[],
        gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
    )


def build_trainer(model, processor, dataset_encoded, data_collator, tracker: MetricsTracker):
    steps_per_epoch = len(dataset_encoded["train"]) // (
        CONFIG.batch_size_train * CONFIG.gradient_accumulation_steps
    )
    args = build_training_args(steps_per_epoch)

    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG.early_stopping_patience,
                early_stopping_threshold=CONFIG.early_stopping_threshold,
            ),
            MetricsCallback(tracker),
        ],
    )
