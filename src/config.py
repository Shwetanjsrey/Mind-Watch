from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

    ORIGINAL_DATASET_DIR = DATA_DIR / "original_dataset"
    PREPROCESSED_DATASET_DIR = DATA_DIR / "preprocessed_dataset"
    REDDIT_CORPUS_DIR = DATA_DIR / "reddit_depression_corpora"

    LABELS = {
        "not depression": 0,
        "moderate": 1,
        "severe": 2
    }

    ID_TO_LABEL = {
        0: "not depression",
        1: "moderate",
        2: "severe"
    }

    MAX_LENGTH = 256
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 5
    RANDOM_SEED = 42

    BASELINE_MODELS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased"
}

    PROPOSED_MODEL = "microsoft/deberta-base"