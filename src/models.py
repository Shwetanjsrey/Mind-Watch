from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    return tokenizer, model