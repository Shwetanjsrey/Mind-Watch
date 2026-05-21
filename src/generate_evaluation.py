import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    print("STARTING EVALUATION...")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

    sys.path.append(CURRENT_DIR)

    from binary_dataset import load_binary_dataset

    MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "binary_mindwatch")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation_outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Model path:", MODEL_PATH)
    print("Output path:", OUTPUT_DIR)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("Loading dataset...")
    _, val_df = load_binary_dataset()

    texts = val_df["text"].tolist()
    labels = np.array(val_df["label"].tolist())

    print("Validation samples:", len(texts))

    probs_all = []

    batch_size = 16

    print("Generating predictions...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]

        probs_all.extend(probs.cpu().numpy())

    probs_all = np.array(probs_all)
    preds = (probs_all >= 0.5).astype(int)

    print("Generating ROC...")
    fpr, tpr, _ = roc_curve(labels, probs_all)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    print("Generating confusion matrix...")
    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Severe", "Severe"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    print("Generating threshold plot...")
    thresholds = np.arange(0.1, 1.0, 0.1)

    accs = []
    precs = []
    recs = []
    f1s = []

    for t in thresholds:
        p = (probs_all >= t).astype(int)

        accs.append(accuracy_score(labels, p))
        precs.append(precision_score(labels, p, zero_division=0))
        recs.append(recall_score(labels, p, zero_division=0))
        f1s.append(f1_score(labels, p, zero_division=0))

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "threshold_performance.png"),
        dpi=300
    )
    plt.close()

    print("Generating error distribution...")
    errors = np.abs(labels - probs_all)

    plt.figure(figsize=(7, 6))
    plt.hist(errors, bins=30)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "prediction_error_distribution.png"),
        dpi=300
    )
    plt.close()

    print("DONE.")
    print("Files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()