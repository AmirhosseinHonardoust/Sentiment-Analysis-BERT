import subprocess
import sys
from pathlib import Path

import pandas as pd

from preprocess import can_stratify


def test_can_stratify_true_when_every_class_has_enough_rows():
    assert can_stratify(["a", "a", "b", "b"], min_per_class=2) is True


def test_can_stratify_false_with_singleton_class():
    assert can_stratify(["a", "a", "b"], min_per_class=2) is False


def test_can_stratify_false_with_single_class():
    assert can_stratify(["a", "a", "a"], min_per_class=2) is False


def test_preprocess_end_to_end_writes_three_csvs(tmp_path: Path):
    src_dir = Path(__file__).resolve().parent.parent / "src"
    input_csv = tmp_path / "sample.csv"
    rows = ["negative"] * 6 + ["neutral"] * 6 + ["positive"] * 6
    df = pd.DataFrame({"text": [f"example text {i}" for i in range(len(rows))], "label": rows})
    df.to_csv(input_csv, index=False)

    outdir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(src_dir / "preprocess.py"),
            "--input",
            str(input_csv),
            "--outdir",
            str(outdir),
            "--val-size",
            "0.2",
            "--test-size",
            "0.2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    train_df = pd.read_csv(outdir / "train.csv")
    val_df = pd.read_csv(outdir / "val.csv")
    test_df = pd.read_csv(outdir / "test.csv")

    assert list(train_df.columns) == ["text", "label"]
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["label"]) == {"negative", "neutral", "positive"}
