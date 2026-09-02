import subprocess
import sys
from pathlib import Path

import pandas as pd


def _tiny_csv(path: Path, n_per_class: int = 4) -> None:
    rows = []
    for label, words in [
        ("negative", "bad terrible awful disappointing"),
        ("neutral", "ok fine average normal"),
        ("positive", "good great awesome excellent"),
    ]:
        for i in range(n_per_class):
            rows.append({"text": f"{words} example {i}", "label": label})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_bert_train_then_evaluate_then_predict_end_to_end(tmp_path: Path, tiny_bert_dir: str):
    """Exercises the BERT train -> evaluate -> predict pipeline fully offline, against
    a tiny randomly-initialized local checkpoint (tiny_bert_dir fixture) instead of a
    real Hub download. Previously this entire path had zero test coverage because it
    required network access to bert-base-uncased; this now runs in CI on every push.
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _tiny_csv(train_csv, n_per_class=4)
    _tiny_csv(val_csv, n_per_class=2)
    _tiny_csv(test_csv, n_per_class=2)

    outdir = tmp_path / "outputs"
    train_result = subprocess.run(
        [
            sys.executable,
            str(src_dir / "train_bert.py"),
            "--train",
            str(train_csv),
            "--val",
            str(val_csv),
            "--outdir",
            str(outdir),
            "--model",
            tiny_bert_dir,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--max-len",
            "16",
        ],
        capture_output=True,
        text=True,
    )
    assert train_result.returncode == 0, train_result.stderr
    assert (outdir / "best_model_bert.pt").exists()
    assert (outdir / "config.json").exists()  # HF format: model.save_pretrained(outdir)

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
            "--max-len",
            "16",
        ],
        capture_output=True,
        text=True,
    )
    assert eval_result.returncode == 0, eval_result.stderr
    assert (outdir / "confusion_matrix.png").exists()
    assert (outdir / "classification_report.txt").exists()

    predict_result = subprocess.run(
        [
            sys.executable,
            str(src_dir / "predict.py"),
            "--checkpoint",
            str(outdir),
            "--max-len",
            "16",
            "--text",
            "good great awesome",
        ],
        capture_output=True,
        text=True,
    )
    assert predict_result.returncode == 0, predict_result.stderr
    assert predict_result.stdout.strip().startswith("[")
