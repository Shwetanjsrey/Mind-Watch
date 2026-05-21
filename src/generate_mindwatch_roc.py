import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

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
    print("Generating ROC for proposed MindWatch model...")

    _, val_df = load_binary_dataset()

    texts = val_df["text"].tolist()
    labels = val_df["label"].tolist()

    y_true = []
    y_scores = []

    total = len(texts)

    for i, text in enumerate(texts):
        clean = safe_text(text)

        result = analyze_risk(clean)

        high_prob = result["high_probability"] / 100.0

        y_true.append(labels[i])
        y_scores.append(high_prob)

        if i % 100 == 0:
            print(f"Processed {i}/{total}")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    print("Computing ROC...")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    output_dir = os.path.join(PROJECT_ROOT, "evaluation_outputs")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"MindWatch AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Proposed MindWatch Prediction Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        "mindwatch_roc_curve.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("\nDONE")
    print("Saved:", save_path)
    print("AUC:", round(roc_auc, 4))


if __name__ == "__main__":
    main()