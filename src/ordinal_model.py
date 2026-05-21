from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config


def load_proposed_model():
    tokenizer = AutoTokenizer.from_pretrained(Config.PROPOSED_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.PROPOSED_MODEL,
        num_labels=3
    )

    return tokenizer, model