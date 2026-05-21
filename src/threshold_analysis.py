import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

from binary_dataset import load_binary_dataset
from risk_engine import analyze_risk


def safe_text(text, max_chars=1200):
    if not isinstance(text, str):
        return ""
    return text[:max_chars]


def main():
    print("Threshold Performance Analysis...")

    _, val_df = load_binary_dataset()

    texts = val_df["text"].tolist()
    labels = np.array(val_df["label"].tolist())

    probs = []

    total = len(texts)

    for i, text in enumerate(texts):
        result = analyze_risk(safe_text(text))

        high_prob = result["high_probability"] / 100.0
        probs.append(high_prob)

        if i % 100 == 0:
            print(f"Processed {i}/{total}")

    probs = np.array(probs)

    thresholds = np.arange(0.1, 1.0, 0.1)

    accs = []
    precs = []
    recs = []
    f1s = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        accs.append(accuracy_score(labels, preds))
        precs.append(precision_score(labels, preds, zero_division=0))
        recs.append(recall_score(labels, preds, zero_division=0))
        f1s.append(f1_score(labels, preds, zero_division=0))

        print(
            f"Threshold {t:.1f} | "
            f"Acc={accs[-1]:.4f} | "
            f"Prec={precs[-1]:.4f} | "
            f"Recall={recs[-1]:.4f} | "
            f"F1={f1s[-1]:.4f}"
        )

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_outputs")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Performance Score")
    plt.title("Prediction Accuracy at Different Tolerance Levels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "threshold_analysis.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("\nDONE")
    print("Saved:", save_path)


if __name__ == "__main__":
    main()