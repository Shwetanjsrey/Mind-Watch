import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from config import Config


def load_original_dataset(split: str):
    file_path = Config.ORIGINAL_DATASET_DIR / f"{split}.tsv"
    df = pd.read_csv(file_path, sep="\t")

    column_map = {
        "train": {"text": "Text_data", "pid": "PID", "label": "Label"},
        "dev": {"text": "Text data", "pid": "PID", "label": "Label"},
        "test": {"text": "text data", "pid": "Pid"}
    }

    cols = column_map[split]

    if split == "test":
        df = df[[cols["pid"], cols["text"]]]
        df.columns = ["pid", "text"]
    else:
        df = df[[cols["pid"], cols["text"], cols["label"]]]
        df.columns = ["pid", "text", "label"]
        df["label"] = df["label"].map(Config.LABELS)

    return df


def create_train_val_split():
    train_df = load_original_dataset("train")
    dev_df = load_original_dataset("dev")

    combined = pd.concat([train_df, dev_df], ignore_index=True)

    combined.drop_duplicates(subset=["text"], inplace=True)

    train_split, val_split = train_test_split(
        combined,
        test_size=0.2,
        stratify=combined["label"],
        random_state=Config.RANDOM_SEED
    )

    return train_split, val_split


def hf_dataset(train_df, val_df):
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])
    return train_ds, val_ds