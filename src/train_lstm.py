"""Train a lightweight BiLSTM sentiment classifier (no pretrained weights needed).

A from-scratch word vocabulary is built from the training split and reused
for validation/evaluation. Writes best_model_lstm.pt (state_dict + vocab +
architecture args) and training_curves.png to --outdir.
"""

import argparse
import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import encode_lstm_text, label_ids, plot_training_curves


class TextDS(Dataset):
    """Builds (or reuses) a word vocab and encodes texts to fixed-length id sequences."""

    def __init__(self, df, vocab=None, max_len=64, build=False, min_freq=1):
        self.texts = df["text"].tolist()
        self.labels = label_ids(df["label"].tolist())
        self.max_len = max_len
        if build or vocab is None:
            self.vocab = {"<pad>": 0, "<unk>": 1}
            cnt = Counter()
            for t in self.texts:
                cnt.update(t.split())
            for w, c in cnt.items():
                if c >= min_freq and w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        else:
            self.vocab = vocab

    def encode(self, s):
        return encode_lstm_text(self.vocab, s, self.max_len)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.encode(self.texts[idx]), dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


class LSTMClassifier(nn.Module):
    """BiLSTM over mean-pooled token embeddings, followed by a linear classifier head."""

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.lstm(e)
        feat = torch.mean(out, dim=1)
        return self.fc(self.drop(feat))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="Path to training CSV (text,label columns).")
    ap.add_argument("--val", required=True, help="Path to validation CSV (text,label columns).")
    ap.add_argument(
        "--outdir", default="outputs", help="Directory to write the checkpoint and curves to."
    )
    ap.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=64, help="Training/validation batch size.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    ap.add_argument("--embed-dim", type=int, default=100, help="Word embedding dimension.")
    ap.add_argument("--hidden-dim", type=int, default=128, help="LSTM hidden state dimension.")
    ap.add_argument(
        "--max-len", type=int, default=64, help="Max token sequence length (truncate/pad to)."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for torch/numpy/random.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tr = pd.read_csv(args.train)
    va = pd.read_csv(args.val)
    trds = TextDS(tr, build=True, max_len=args.max_len, min_freq=1)
    vads = TextDS(va, vocab=trds.vocab, max_len=args.max_len)

    train_loader = DataLoader(trds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(vads, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(len(trds.vocab), args.embed_dim, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = 1e9
    # Namespaced so it never collides with train_bert.py's checkpoint in the
    # same --outdir (both used to write "best_model.pt").
    best_path = os.path.join(args.outdir, "best_model_lstm.pt")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
            n += xb.size(0)
        tr_loss = tl / n

        model.eval()
        vl = 0
        vn = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {ep}/{args.epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vl += loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vl / vn
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        print(f"[epoch {ep}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab": trds.vocab,
                    "max_len": args.max_len,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "model_type": "lstm",
                },
                best_path,
            )
        plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))

    print("[OK] Training complete. Best saved to", best_path)


if __name__ == "__main__":
    main()
