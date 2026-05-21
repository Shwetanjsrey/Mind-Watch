from transformers import (
    TrainingArguments,
    Trainer
)

import torch
import gc

from dataset import hf_dataset
from metrics import compute_metrics
from models import load_model
from config import Config
from seed import set_seed


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH
    )


def train_model(train_df, val_df, model_name, run_name):
    print(f"\nStarting training for: {run_name}")

    set_seed(Config.RANDOM_SEED)

    tokenizer, model = load_model(model_name)

    train_ds, val_ds = hf_dataset(train_df, val_df)

    train_ds = train_ds.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True
    )

    val_ds = val_ds.map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True
    )

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    args = TrainingArguments(
        output_dir=f"./outputs/{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="./outputs/logs",
        report_to="none",
        fp16=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print(f"\nFinal Evaluation Results for {run_name}:")
    for key, value in results.items():
        print(f"{key}: {value}")

    del trainer
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return results