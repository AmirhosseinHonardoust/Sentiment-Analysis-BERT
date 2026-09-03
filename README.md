<div align="center">
    
# Sentiment Analysis with BERT

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-BERT%20%2B%20BiLSTM-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Portfolio%20ML%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-BERT/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-BERT/actions/workflows/ci.yml)

</div>

A deep learning project for **sentiment classification of tweets**, built around fine-tuning **BERT (`bert-base-uncased`)** with a lightweight **BiLSTM baseline** for comparison, plus reproducible **preprocessing**, **evaluation reporting**, and **single-text inference**.

> **Note on data:** `data/tweets_sample.csv` is a ~15-row illustrative sample, not a real training set. It is enough to exercise the full pipeline end-to-end, but the metrics and plots checked into `outputs/` are demo artifacts, not a benchmark of real model quality. Swap in a real labeled dataset (same `text,label` schema) for meaningful results.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Preprocess Data](#preprocess-data)
- [Train BERT](#train-bert)
- [Train LSTM](#train-lstm)
- [Evaluate](#evaluate)
- [Predict](#predict)
- [Results](#results)
- [Development and Testing](#development-and-testing)
- [Requirements](#requirements)
- [Known Limitations](#known-limitations)
- [Next Steps](#next-steps)
- [Author](#author)
- [License](#license)

---

## Project Overview

Sentiment classification is often demonstrated as a single accuracy number on a clean dataset, which hides most of the engineering that makes a model usable in practice: consistent preprocessing, a reproducible train/val/test split, an evaluation report that generalizes across model types, and a way to sanity-check predictions on raw text.

This project fine-tunes a transformer model on tweet sentiment (`negative`, `neutral`, `positive`) and pairs it with a lightweight BiLSTM baseline trained from scratch, so the two can be compared on identical data through the same evaluation entry point. It includes preprocessing, training for both model types, classification reports, confusion matrices, ROC curves, word clouds, and a CI-tested, offline-friendly test suite.

The goal is to show a modular, reproducible NLP pipeline rather than a single leaderboard score.

---

## What This Project Does

This project can:

- Clean, tokenize, and split raw tweets into train/val/test sets
- Fine-tune `bert-base-uncased` on sentiment labels
- Train a BiLSTM baseline as a lightweight, no-pretrained-weights alternative
- Evaluate either model through the same `evaluate.py` entry point, with checkpoint type auto-detected
- Generate classification reports, confusion matrices, and ROC curves
- Generate word clouds for positive and negative predictions
- Track and visualize training and validation loss
- Classify one or more raw text strings directly, without a labeled CSV
- Run a full offline test suite in CI, including the BERT code path, without a live model download

---

## Key Features

- **Shared preprocessing pipeline** producing consistent `train.csv` / `val.csv` / `test.csv` splits
- **BERT fine-tuning** via Hugging Face Transformers (`bert-base-uncased`)
- **BiLSTM baseline** for a fast, pretrained-weight-free comparison
- **Unified evaluation**, `evaluate.py` auto-detects BERT vs. LSTM checkpoints
- **Classification reports, confusion matrices, and ROC curves** for either model
- **Word cloud generation** for positive and negative predictions
- **Single- or multi-text inference** with `predict.py`, same checkpoint auto-detection
- **Offline CI test suite**, including a tiny randomly-initialized local BERT checkpoint fixture (`tests/conftest.py`) so no network access or real download is required
- **Lint/format/type-check gate**, Ruff, Black, and mypy, enforced in CI alongside pytest
- **Dependabot-managed dependencies** with a required full CI pass on every bump

---

## System Workflow

```text
Raw tweets (text, label)
        ↓
Preprocessing (clean, tokenize, split)
        ↓
train.csv / val.csv / test.csv
        ↓
BERT fine-tuning   or   BiLSTM baseline
        ↓
Evaluation (classification report, confusion matrix, ROC curve)
        ↓
Word clouds and training curves
        ↓
Single-text prediction (predict.py)
```

---

## Project Structure

```text
sentiment-analysis-bert/
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml
│   └── dependabot.yml
│
├── data/
│   ├── tweets_sample.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── outputs/
│   ├── best_model_bert.pt
│   ├── best_model_lstm.pt
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── classification_report.txt
│   ├── wordcloud_negative.png
│   └── roc_curve.png
│
├── src/
│   ├── preprocess.py
│   ├── train_lstm.py
│   ├── train_bert.py
│   ├── bert_dataset.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│   └── conftest.py
│
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-BERT.git
cd Sentiment-Analysis-BERT
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (Ruff, Black, mypy, pytest):

```bash
pip install -r requirements-dev.txt
```

---

## Preprocess Data

Cleans and tokenizes the raw tweets, then splits them into train/val/test CSVs with the same `text,label` schema throughout.

```bash
python src/preprocess.py --input data/tweets_sample.csv --outdir data --val-size 0.15 --test-size 0.15
```

---

## Train BERT

Fine-tunes `bert-base-uncased` on the training split, validating against `val.csv`.

```bash
python src/train_bert.py --train data/train.csv --val data/val.csv --outdir outputs/bert --epochs 3 --batch-size 16 --lr 2e-5 --model bert-base-uncased --max-len 128
```

---

## Train LSTM

A lightweight BiLSTM baseline that needs no pretrained weights, useful for fast iteration or as a comparison point against BERT.

```bash
python src/train_lstm.py --train data/train.csv --val data/val.csv --outdir outputs/lstm --epochs 8 --batch-size 64 --lr 1e-3
```

---

## Evaluate

Works for either model, `evaluate.py` auto-detects the checkpoint type from `--checkpoint`.

```bash
# BERT: point at the HF-format output directory (contains config.json)
python src/evaluate.py --test data/test.csv --checkpoint outputs/bert --outdir outputs/bert --wordclouds

# LSTM: point at the checkpoint file, or its containing directory
python src/evaluate.py --test data/test.csv --checkpoint outputs/lstm --outdir outputs/lstm --wordclouds
```

Generated outputs include:

```text
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/roc_curve.png
outputs/training_curves.png
outputs/wordcloud_negative.png
```

---

## Predict

Classify one or more texts directly, without a labeled CSV. Uses the same checkpoint auto-detection as `evaluate.py`.

```bash
python src/predict.py --checkpoint outputs/lstm --text "I love this!" --text "Broke after a week."
```

---

## Results

- **Classification Report:** `outputs/classification_report.txt`

<div align="center">

| Confusion Matrix | ROC Curve |
|---|---|
| ![Confusion matrix](outputs/confusion_matrix.png) | ![ROC curve](outputs/roc_curve.png) |
| **Analysis:** Shows correct vs. incorrect predictions across the three sentiment classes on the held-out test set. | **Analysis:** Shows class separation across decision thresholds, read alongside the small sample size noted below. |

</div>

<div align="center">

| Training Curves | Negative Wordcloud |
|---|---|
| ![Training curves](outputs/training_curves.png) | ![Negative wordcloud](outputs/wordcloud_negative.png) |
| **Analysis:** Training and validation loss over epochs, useful for spotting over/underfitting. | **Analysis:** Most frequent terms in negative predictions, generated with `--wordclouds`. |

</div>

> The checked-in metrics and plots were generated from the ~15-row `tweets_sample.csv` demo data. They confirm the pipeline runs end-to-end, not that the model performs well, see [Known Limitations](#known-limitations).
>
> Image paths above are relative to the repo root (`outputs/...`) and render on GitHub; they won't resolve if this file is viewed outside the repository.

---

## Development and Testing

These four commands are exactly what CI runs on every push and pull request:

```bash
ruff check --select E,F,I,B,SIM,UP --line-length 100 src/ tests/
black --check --line-length 100 src/ tests/
mypy --ignore-missing-imports src/
pytest
```

The pytest suite covers preprocessing, the LSTM training/eval pipeline, utility functions, and the full BERT train/evaluate/predict pipeline, all without network access, since the BERT tests run against a tiny, randomly-initialized local checkpoint (`tests/conftest.py`'s `tiny_bert_dir` fixture) rather than a real download. See `CONTRIBUTING.md` for details.

> A real fine-tuning run against `bert-base-uncased` still needs a live Hugging Face Hub download and is exercised manually rather than in CI.

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Requirements

- Python 3.11+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn, pandas, matplotlib, seaborn
- WordCloud
- pytest, Ruff, Black, mypy
- GitHub Actions

---

## Known Limitations

This project has important limitations:

- `data/tweets_sample.csv` is a 15-row illustrative sample, don't read anything into the demo metrics/plots in `outputs/` beyond "the pipeline runs end-to-end"
- BERT's *code path* (training loop, evaluation, inference) is tested in CI against a tiny local checkpoint, but a real fine-tuning run against `bert-base-uncased` requires a live Hugging Face Hub download and is only exercised manually
- Pinned dependencies are refreshed periodically via Dependabot (`.github/dependabot.yml`); each bump PR should pass the full gate, including the BERT tests, before merging

The project is strongest as a portfolio demonstration of a modular, reproducible transformer fine-tuning workflow, not as a benchmark of real-world sentiment accuracy.

---

## Next Steps

Potential next improvements:

- Expand the dataset for more robust evaluation
- Try advanced transformer models (RoBERTa, DistilBERT)
- Apply hyperparameter tuning and cross-validation
- Deploy the model with FastAPI or Streamlit for an interactive demo

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
