from transformers import pipeline


emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0
)


def analyze_emotions(text):
    results = emotion_classifier(text)[0]

    filtered = {
        item["label"]: round(item["score"], 4)
        for item in results
        if item["score"] > 0.1
    }

    return filtered