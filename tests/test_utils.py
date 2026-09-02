import json
from pathlib import Path

import numpy as np

from utils import clean_text, plot_confusion_matrix, plot_roc, save_metrics


def test_clean_text_lowercases_and_strips_urls_mentions_hashtags():
    raw = "Check THIS out https://example.com/x @someone #GreatStuff!!"
    cleaned = clean_text(raw)
    assert cleaned == "check this out greatstuff"


def test_clean_text_collapses_whitespace_and_newlines():
    assert clean_text("hello\n\n\tworld   again") == "hello world again"


def test_clean_text_keeps_apostrophes():
    assert clean_text("It's great") == "it's great"


def test_plot_confusion_matrix_default_labels(tmp_path: Path):
    outpath = tmp_path / "cm.png"
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    plot_confusion_matrix(y_true, y_pred, str(outpath))
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_confusion_matrix_restricted_labels(tmp_path: Path):
    """Only classes present in the data should be usable (fixes the previous
    behavior of always drawing all 3 classes regardless of what's present)."""
    outpath = tmp_path / "cm_subset.png"
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # Should not raise even though class 2 never appears.
    plot_confusion_matrix(y_true, y_pred, str(outpath), labels=[0, 1])
    assert outpath.exists() and outpath.stat().st_size > 0


def test_plot_roc_returns_macro_auc_in_valid_range(tmp_path: Path):
    outpath = tmp_path / "roc.png"
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
            [0.1, 0.3, 0.6],
        ]
    )
    macro_auc = plot_roc(y_true, probs, str(outpath))
    assert outpath.exists()
    assert 0.0 <= macro_auc <= 1.0


def test_plot_roc_with_restricted_labels(tmp_path: Path):
    outpath = tmp_path / "roc_subset.png"
    y_true = [0, 0, 1, 1]
    probs = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    macro_auc = plot_roc(y_true, probs, str(outpath), labels=[0, 1])
    assert outpath.exists()
    assert 0.0 <= macro_auc <= 1.0


def test_save_metrics_writes_report_and_json(tmp_path: Path):
    save_metrics("dummy report text", 0.87, str(tmp_path))
    report_path = tmp_path / "classification_report.txt"
    metrics_path = tmp_path / "metrics.json"
    assert report_path.read_text() == "dummy report text"
    assert json.loads(metrics_path.read_text()) == {"macro_roc_auc": 0.87}
