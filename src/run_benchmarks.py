from dataset import create_train_val_split
from trainer import train_model
from config import Config


def main():
    train_df, val_df = create_train_val_split()

    for name, model_path in Config.BASELINE_MODELS.items():
        print(f"\nTraining {name}...")
        train_model(
            train_df=train_df,
            val_df=val_df,
            model_name=model_path,
            run_name=name
        )


if __name__ == "__main__":
    main()