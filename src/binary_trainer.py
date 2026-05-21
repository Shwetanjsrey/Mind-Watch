from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import gc

from metrics import compute_metrics
from config import Config
from seed import set_seed
from binary_dataset import hf_binary_dataset


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH
    )


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = torch.tensor(
            [0.56, 4.73],
            dtype=torch.float
        ).to(model.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_binary_model(train_df, val_df, model_name, run_name):
    print(f"\nTraining binary model: {run_name}")

    set_seed(Config.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    train_ds, val_ds = hf_binary_dataset(train_df, val_df)

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
        learning_rate=1e-5,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    trainer.save_model("./outputs/binary_mindwatch")
    tokenizer.save_pretrained("./outputs/binary_mindwatch")

    print("\nFinal Severe Risk Detector Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    del trainer
    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return results