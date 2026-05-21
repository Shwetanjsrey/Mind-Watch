from dataset import create_train_val_split
from trainer import train_model
from config import Config


def main():
    train_df, val_df = create_train_val_split()

    train_model(
        train_df=train_df,
        val_df=val_df,
        model_name=Config.PROPOSED_MODEL,
        run_name="proposed_deberta"
    )


if __name__ == "__main__":
    main()