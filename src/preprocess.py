"""Split a raw text,label CSV into train/val/test CSVs, stratifying by label when possible.

Falls back to an unstratified split for small/imbalanced inputs where every
class doesn't have enough rows to stratify (see can_stratify).
"""

import argparse
import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


def can_stratify(labels, min_per_class=2):
    """True if every class in `labels` has at least `min_per_class` rows and there's >1 class."""
    counts = Counter(labels)
    return all(c >= min_per_class for c in counts.values()) and len(counts) > 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the raw input CSV (text,label).")
    ap.add_argument("--outdir", default="data", help="Directory to write train/val/test CSVs to.")
    ap.add_argument(
        "--val-size", type=float, default=0.15, help="Fraction of rows to hold out for val."
    )
    ap.add_argument(
        "--test-size", type=float, default=0.15, help="Fraction of rows to hold out for test."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for the splits.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")

    # Clean trivial empties
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    total_split = args.val_size + args.test_size
    if total_split <= 0 or total_split >= 1:
        raise ValueError("--val-size + --test-size must be in (0,1).")

    # First split: train vs temp (val+test)
    use_strat1 = can_stratify(df["label"])
    if use_strat1:
        train_df, temp = train_test_split(
            df, test_size=total_split, stratify=df["label"], random_state=args.seed
        )
    else:
        # Fallback without stratify for tiny/imbalanced data
        train_df, temp = train_test_split(df, test_size=total_split, random_state=args.seed)

    # Second split: val vs test from temp
    rel_test = args.test_size / total_split
    use_strat2 = can_stratify(temp["label"])
    if use_strat2:
        val_df, test_df = train_test_split(
            temp, test_size=rel_test, stratify=temp["label"], random_state=args.seed
        )
    else:
        val_df, test_df = train_test_split(temp, test_size=rel_test, random_state=args.seed)

    # Safety: ensure each split has at least one sample of each class, otherwise warn
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = Counter(part["label"])
        if any(c == 0 for c in counts.values()) or len(counts) < len(Counter(df["label"])):
            print(
                f"[WARN] '{name}' split may be missing some classes. "
                "Consider adding more data or reducing --val-size/--test-size."
            )

    train_df.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.outdir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.outdir, "test.csv"), index=False)
    print("[OK] Wrote train/val/test CSVs to", args.outdir)


if __name__ == "__main__":
    main()
