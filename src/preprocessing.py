import re
from dataset import create_train_val_split
from config import Config


def clean_text(text: str):
    text = str(text)

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_and_save():
    train_df, val_df = create_train_val_split()

    train_df["text"] = train_df["text"].apply(clean_text)
    val_df["text"] = val_df["text"].apply(clean_text)

    train_df.to_csv(Config.PREPROCESSED_DATASET_DIR / "train_clean.csv", index=False)
    val_df.to_csv(Config.PREPROCESSED_DATASET_DIR / "val_clean.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")


if __name__ == "__main__":
    preprocess_and_save()