"""Evaluate a trained BERT or LSTM checkpoint against a labeled test CSV.

Auto-detects the checkpoint type from --checkpoint: an HF model directory
(config.json present) is evaluated as BERT, otherwise as an LSTM checkpoint
(best_model_lstm.pt or its containing directory). Writes a classification
report, confusion matrix, ROC curve, metrics.json, and optionally wordclouds
to --outdir.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from utils import (
    ID2LABEL,
    encode_lstm_text,
    label_ids,
    plot_confusion_matrix,
    plot_roc,
    save_metrics,
)

# ---------------- Shared ---------------- #


def is_hf_dir(path: str) -> bool:
    """True if `path` is a directory containing an HF `config.json` (a BERT checkpoint)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _run_wordclouds(df: pd.DataFrame, y_pred: np.ndarray, outdir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud

        for cls, name in [(0, "negative"), (2, "positive")]:
            texts = " ".join(
                t
                for t, pred in zip(df["text"].tolist(), y_pred.tolist(), strict=True)
                if pred == cls
            )
            if not texts.strip():
                continue
            wc = WordCloud(width=800, height=400, background_color="white").generate(texts)
            plt.figure(figsize=(8, 4))
            plt.imshow(wc)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"wordcloud_{name}.png"), dpi=160)
            plt.close()
    except Exception as e:
        print("[WARN] Wordcloud generation skipped:", e)


def _score_and_report(
    y_true: List[int], y_prob: np.ndarray, df: pd.DataFrame, outdir: str, wordclouds: bool
) -> None:
    """Shared scoring/plotting path used by both the BERT and LSTM evaluators."""
    y_pred = y_prob.argmax(axis=1)

    # Dynamically adapt to classes present in y_true, rather than assuming all 3.
    labels_present: List[int] = sorted(np.unique(y_true).tolist())
    target_names = [ID2LABEL.get(i, str(i)) for i in labels_present]

    report = classification_report(
        y_true, y_pred, labels=labels_present, target_names=target_names, zero_division=0
    )
    print(report)

    # Confusion matrix and ROC now both respect labels_present (previously the
    # confusion matrix was drawn against the full 3-class set regardless).
    plot_confusion_matrix(
        y_true, y_pred, os.path.join(outdir, "confusion_matrix.png"), labels=labels_present
    )
    macro_roc_auc = plot_roc(
        y_true, y_prob, os.path.join(outdir, "roc_curve.png"), labels=labels_present
    )

    save_metrics(report, macro_roc_auc, outdir)

    if wordclouds:
        _run_wordclouds(df, y_pred, outdir)

    print("[OK] Evaluation complete. Outputs saved to", outdir)


# ---------------- BERT ---------------- #


def evaluate_bert(
    test_csv: str, ckpt_dir: str, outdir: str, max_len: int = 128, wordclouds: bool = False
) -> None:
    """Evaluate an HF-format BERT checkpoint directory against `test_csv`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from bert_dataset import BertTweetDataset

    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)

    df = pd.read_csv(test_csv)
    ds = BertTweetDataset(df, tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    y_true: List[int] = []
    y_prob_parts = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            y_prob_parts.append(probs)
            y_true.extend(batch["labels"].cpu().numpy().tolist())
    y_prob = np.concatenate(y_prob_parts, axis=0)

    _score_and_report(y_true, y_prob, df, outdir, wordclouds)


# ---------------- LSTM ---------------- #


def find_lstm_checkpoint(path: str) -> str:
    """`path` may be the checkpoint file itself or a directory containing it."""
    if os.path.isfile(path):
        return path
    candidate = os.path.join(path, "best_model_lstm.pt")
    if os.path.isfile(candidate):
        return candidate
    raise ValueError(
        f"No LSTM checkpoint found at '{path}'. Expected a file, or a directory "
        "containing 'best_model_lstm.pt' (as written by train_lstm.py)."
    )


def is_lstm_checkpoint(path: str) -> bool:
    """True if `path` (or `<path>/best_model_lstm.pt`) is an LSTM checkpoint file."""
    if os.path.isfile(path) and path.endswith(".pt"):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return False
        return isinstance(ckpt, dict) and ckpt.get("model_type") == "lstm"
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, "best_model_lstm.pt"))
    return False


class LSTMEvalDS(Dataset):
    """Mirrors train_lstm.py's TextDS encoding (via utils.encode_lstm_text), fixed vocab."""

    def __init__(self, df: pd.DataFrame, vocab: Dict[str, int], max_len: int) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = label_ids(df["label"].tolist())
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple:
        ids = encode_lstm_text(self.vocab, self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate_lstm(test_csv: str, ckpt_path: str, outdir: str, wordclouds: bool = False) -> None:
    """Evaluate an LSTM checkpoint (file or its containing directory) against `test_csv`."""
    # Local import: keeps the LSTM path usable without transformers installed.
    from train_lstm import LSTMClassifier

    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = find_lstm_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]
    # Fall back to train_lstm.py's own defaults for checkpoints saved before
    # embed_dim/hidden_dim were recorded.
    embed_dim = ckpt.get("embed_dim", 100)
    hidden_dim = ckpt.get("hidden_dim", 128)

    model = LSTMClassifier(len(vocab), embed_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    df = pd.read_csv(test_csv)
    ds = LSTMEvalDS(df, vocab, max_len)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    y_true: List[int] = []
    y_prob_parts = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob_parts.append(probs)
            y_true.extend(yb.numpy().tolist())
    y_prob = np.concatenate(y_prob_parts, axis=0)

    _score_and_report(y_true, y_prob, df, outdir, wordclouds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", required=True, help="Path to test CSV (text,label columns).")
    ap.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "For BERT: directory with model files (config.json, tokenizer.json). "
            "For LSTM: the best_model_lstm.pt file, or its containing directory."
        ),
    )
    ap.add_argument("--outdir", default="outputs", help="Directory to write evaluation outputs.")
    ap.add_argument(
        "--max-len", type=int, default=128, help="Max token sequence length (BERT path only)."
    )
    ap.add_argument(
        "--wordclouds", action="store_true", help="Also generate negative/positive wordclouds."
    )
    args = ap.parse_args()

    if is_hf_dir(args.checkpoint):
        evaluate_bert(
            args.test,
            args.checkpoint,
            args.outdir,
            max_len=args.max_len,
            wordclouds=args.wordclouds,
        )
    elif is_lstm_checkpoint(args.checkpoint):
        evaluate_lstm(args.test, args.checkpoint, args.outdir, wordclouds=args.wordclouds)
    else:
        raise ValueError(
            "--checkpoint must be either a BERT model directory (containing "
            "config.json) or an LSTM checkpoint (best_model_lstm.pt / its directory)."
        )


if __name__ == "__main__":
    main()
