from binary_dataset import load_binary_dataset
from binary_trainer import train_binary_model
from config import Config


def main():
    train_df, val_df = load_binary_dataset()

    train_binary_model(
        train_df,
        val_df,
        Config.PROPOSED_MODEL,
        "binary_mindwatch"
    )


if __name__ == "__main__":
    main()