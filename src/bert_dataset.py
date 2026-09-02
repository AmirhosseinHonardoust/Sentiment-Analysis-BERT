"""Shared BERT tokenized dataset, used by both train_bert.py and evaluate.py.

Deliberately imports only torch (already a hard dependency of the LSTM path
too), not transformers — the tokenizer is passed in by the caller — so this
module stays importable without pulling in transformers for callers that
don't need it.
"""

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import label_ids


class BertTweetDataset(Dataset):
    """Tokenizes `df["text"]` with `tokenizer` and encodes `df["label"]` via LABEL2ID."""

    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_len: int = 128) -> None:
        self.texts = df["text"].astype(str).tolist()
        self.labels = label_ids(df["label"].tolist())
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item
