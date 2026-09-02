import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from evaluate import is_hf_dir, is_lstm_checkpoint


def test_is_hf_dir_requires_config_json(tmp_path: Path):
    assert is_hf_dir(str(tmp_path)) is False
    (tmp_path / "config.json").write_text("{}")
    assert is_hf_dir(str(tmp_path)) is True


def test_is_lstm_checkpoint_false_for_non_lstm_dict(tmp_path: Path):
    import torch

    ckpt_path = tmp_path / "best_model_bert.pt"
    torch.save({"model_type": "bert", "model_name": "bert-base-uncased"}, ckpt_path)
    assert is_lstm_checkpoint(str(ckpt_path)) is False


def test_is_lstm_checkpoint_true_for_lstm_dict(tmp_path: Path):
    import torch

    ckpt_path = tmp_path / "best_model_lstm.pt"
    torch.save({"model_type": "lstm", "vocab": {"<pad>": 0}}, ckpt_path)
    assert is_lstm_checkpoint(str(ckpt_path)) is True
    # Also resolvable by directory.
    assert is_lstm_checkpoint(str(tmp_path)) is True


def _tiny_csv(path: Path, n_per_class: int = 4):
    rows = []
    for label, words in [
        ("negative", "bad terrible awful disappointing"),
        ("neutral", "ok fine average normal"),
        ("positive", "good great awesome excellent"),
    ]:
        for i in range(n_per_class):
            rows.append({"text": f"{words} example {i}", "label": label})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_lstm_train_then_evaluate_end_to_end(tmp_path: Path):
    """Exercises the new LSTM evaluation path this PR adds: previously
    train_lstm.py had no corresponding evaluator at all."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _tiny_csv(train_csv, n_per_class=6)
    _tiny_csv(val_csv, n_per_class=2)
    _tiny_csv(test_csv, n_per_class=2)

    outdir = tmp_path / "outputs"
    train_result = subprocess.run(
        [
            sys.executable,
            str(src_dir / "train_lstm.py"),
            "--train",
            str(train_csv),
            "--val",
            str(val_csv),
            "--outdir",
            str(outdir),
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
    )
    assert train_result.returncode == 0, train_result.stderr
    assert (outdir / "best_model_lstm.pt").exists()

    eval_result = subprocess.run(
        [
            sys.executable,
            str(src_dir / "evaluate.py"),
            "--test",
            str(test_csv),
            "--checkpoint",
            str(outdir),
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
    )
    assert eval_result.returncode == 0, eval_result.stderr
    assert (outdir / "confusion_matrix.png").exists()
    assert (outdir / "roc_curve.png").exists()
    assert (outdir / "classification_report.txt").exists()

    metrics = json.loads((outdir / "metrics.json").read_text())
    assert "macro_roc_auc" in metrics
