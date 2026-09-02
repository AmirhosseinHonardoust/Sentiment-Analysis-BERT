import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#(\w+)", r"\1", s)  # drop hash
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def plot_confusion_matrix(y_true, y_pred, outpath, labels=None):
    """Plot a confusion matrix restricted to `labels` (defaults to all 3 classes)."""
    if labels is None:
        labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = [ID2LABEL.get(i, str(i)) for i in labels]
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_training_curves(history, outpath):
    plt.figure(figsize=(6, 4))
    for k, v in history.items():
        if k.endswith("_loss"):
            plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_roc(y_true, probs, outpath, labels=None):
    """One-vs-rest ROC restricted to `labels` (defaults to all 3 classes).

    `probs` must have one column per class (0..num_classes-1); `labels` selects
    which class columns to plot and average, so callers can pass only the
    classes actually present in `y_true`.
    """
    if labels is None:
        labels = [0, 1, 2]
    ys = pd.get_dummies(y_true).reindex(columns=labels, fill_value=0).values
    aucs = []
    plt.figure(figsize=(5, 4))
    for col, i in enumerate(labels):
        fpr, tpr, _ = roc_curve(ys[:, col], probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{ID2LABEL.get(i, str(i))} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (OvR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return float(np.mean(aucs)) if aucs else 0.0


def save_metrics(report, macro_roc_auc, outdir):
    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"macro_roc_auc": macro_roc_auc}, f, indent=2)
