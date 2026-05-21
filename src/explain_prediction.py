import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./outputs/binary_mindwatch"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

model.eval()


def get_severe_probability(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][1].item()


def explain_text(text):
    words = text.split()

    baseline = get_severe_probability(text)

    contributions = []

    for i, word in enumerate(words):
        modified = words[:i] + words[i+1:]
        modified_text = " ".join(modified)

        score = get_severe_probability(modified_text)

        impact = baseline - score

        contributions.append((word, impact))

    contributions.sort(key=lambda x: x[1], reverse=True)

    top_words = [word for word, impact in contributions[:5]]

    return top_words