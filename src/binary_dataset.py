import pandas as pd
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from config import Config


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,']", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def map_binary(label):
    if label == "severe":
        return 1
    return 0


def oversample_minority(df):
    majority = df[df["label"] == 0]
    minority = df[df["label"] == 1]

    minority_upsampled = minority.sample(
        n=len(majority),
        replace=True,
        random_state=Config.RANDOM_SEED
    )

    balanced = pd.concat([majority, minority_upsampled])
    balanced = balanced.sample(frac=1, random_state=Config.RANDOM_SEED)

    return balanced


def load_binary_dataset():
    train_path = Config.ORIGINAL_DATASET_DIR / "train.tsv"
    dev_path = Config.ORIGINAL_DATASET_DIR / "dev.tsv"

    train_df = pd.read_csv(train_path, sep="\t")
    dev_df = pd.read_csv(dev_path, sep="\t")

    train_df = train_df[["PID", "Text_data", "Label"]]
    train_df.columns = ["pid", "text", "label"]

    dev_df = dev_df[["PID", "Text data", "Label"]]
    dev_df.columns = ["pid", "text", "label"]

    combined = pd.concat([train_df, dev_df], ignore_index=True)

    combined["text"] = combined["text"].apply(clean_text)

    combined.drop_duplicates(subset=["text"], inplace=True)

    combined["label"] = combined["label"].apply(map_binary)

    train_split, val_split = train_test_split(
        combined,
        test_size=0.2,
        stratify=combined["label"],
        random_state=Config.RANDOM_SEED
    )

    train_split = oversample_minority(train_split)

    return train_split, val_split


def hf_binary_dataset(train_df, val_df):
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    return train_ds, val_ds