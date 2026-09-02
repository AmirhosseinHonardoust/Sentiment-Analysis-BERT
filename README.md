# Sentiment Analysis with BERT
 
![CI](https://github.com/AmirhosseinHonardoust/Sentiment-Analysis-BERT/actions/workflows/ci.yml/badge.svg)

A deep learning project for **sentiment classification of tweets** using **BERT (Bidirectional Encoder Representations from Transformers)**, with a lightweight **BiLSTM baseline** for comparison. The project includes data preprocessing, vocabulary/tokenizer setup, model training, evaluation, and visualization of results such as confusion matrix, ROC curves, and word clouds.

> **Note on data:** `data/tweets_sample.csv` is a ~15-row illustrative sample, not a real training set — it's enough to exercise the full pipeline end-to-end, but the metrics/plots checked into `outputs/` are demo artifacts, not a benchmark of real model quality. Swap in a real labeled dataset (same `text,label` schema) for meaningful results.

---

## Features
- Preprocess tweets with cleaning, tokenization, and splitting into train/val/test sets.  
- Fine-tune `bert-base-uncased` **or** train a BiLSTM baseline on sentiment labels (`negative`, `neutral`, `positive`).  
- Track and visualize **training & validation loss**.  
- Generate **classification reports, confusion matrices, ROC curves** for either model via the same `evaluate.py` entry point.
- Create **word clouds** for positive and negative predictions.  
- Modular codebase with reproducible pipelines for preprocessing, training, and evaluation.
- Lint/format/type-check gate (ruff, black, mypy) and a pytest suite, enforced in CI.

---

## Project Structure
```
sentiment-analysis-bert/
├─ .github/workflows/
│  └─ ci.yml
├─ data/
│  ├─ train.csv
│  ├─ val.csv
│  └─ test.csv
├─ outputs/
│  ├─ best_model_bert.pt      # written by train_bert.py
│  ├─ best_model_lstm.pt      # written by train_lstm.py (separate outdir recommended)
│  ├─ confusion_matrix.png
│  ├─ training_curves.png
│  ├─ classification_report.txt
│  ├─ wordcloud_negative.png
│  └─ roc_curve.png
├─ src/
│  ├─ preprocess.py
│  ├─ train_lstm.py
│  ├─ train_bert.py
│  ├─ bert_dataset.py     # shared tokenized dataset for train_bert.py / evaluate.py
│  ├─ evaluate.py
│  ├─ predict.py          # single-text inference, either model
│  └─ utils.py
├─ tests/
├─ pyproject.toml
├─ requirements.txt
├─ requirements-dev.txt
└─ README.md
```
---

## Setup
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

pip install -r requirements.txt

# For contributing (lint/format/type-check tools + pytest):
pip install -r requirements-dev.txt
```

## Preprocess Data
```bash
python src/preprocess.py --input data/tweets_sample.csv --outdir data --val-size 0.15 --test-size 0.15
```

## Train BERT
```bash
python src/train_bert.py --train data/train.csv --val data/val.csv --outdir outputs/bert --epochs 3 --batch-size 16 --lr 2e-5 --model bert-base-uncased --max-len 128
```

## Train LSTM (lightweight baseline, no pretrained weights needed)
```bash
python src/train_lstm.py --train data/train.csv --val data/val.csv --outdir outputs/lstm --epochs 8 --batch-size 64 --lr 1e-3
```

## Evaluate
Works for either model — `evaluate.py` auto-detects the checkpoint type from `--checkpoint`.
```bash
# BERT: point at the HF-format output directory (contains config.json)
python src/evaluate.py --test data/test.csv --checkpoint outputs/bert --outdir outputs/bert --wordclouds

# LSTM: point at the checkpoint file, or its containing directory
python src/evaluate.py --test data/test.csv --checkpoint outputs/lstm --outdir outputs/lstm --wordclouds
```

## Predict
Classify one or more texts directly, without a labeled CSV. Same checkpoint auto-detection as `evaluate.py`.
```bash
python src/predict.py --checkpoint outputs/lstm --text "I love this!" --text "Broke after a week."
```

## Development
```bash
ruff check --select E,F,I,B,SIM,UP --line-length 100 src/ tests/
black --check --line-length 100 src/ tests/
mypy --ignore-missing-imports src/
pytest
```
These four commands are exactly what CI runs on every push/PR. The pytest suite covers preprocessing, the LSTM training/eval pipeline, and utility functions without any network access; the BERT training/eval path requires downloading `bert-base-uncased` from the Hugging Face Hub and is exercised manually / in your own environment rather than in CI.
---

## Results
- **Classification Report:** `outputs/classification_report.txt`
---

- **Confusion Matrix:**  

  <img width="800" height="640" alt="confusion_matrix" src="https://github.com/user-attachments/assets/f7c6d7c0-bb1d-467c-819a-2999a6d32e4f" />
---

- **Training Curves:**  

  <img width="960" height="640" alt="training_curves" src="https://github.com/user-attachments/assets/219af181-15fd-4f1e-aabe-c3234a816af7" />
---

- **Negative Wordcloud:**  

  <img width="1280" height="640" alt="wordcloud_negative" src="https://github.com/user-attachments/assets/1f68aed5-ad15-4388-b132-026fc7e5b65b" />
---

## Requirements
- Python 3.8+  
- PyTorch  
- Transformers (HuggingFace)  
- Scikit-learn, Pandas, Matplotlib, Seaborn  
- WordCloud  

---

## Known limitations
- `data/tweets_sample.csv` is a 15-row illustrative sample; don't read anything into the demo metrics/plots in `outputs/` beyond "the pipeline runs end-to-end."
- Pinned dependency versions (`requirements.txt`) are from mid-2024; a version bump (torch, transformers) hasn't been done yet since it needs re-verifying the BERT path against a real Hugging Face Hub download, which isn't exercised in CI.
- BERT training/eval requires downloading `bert-base-uncased` from the Hugging Face Hub and is not covered by CI or automated tests — only exercised manually.

## Next Steps
- Expand dataset for more robust evaluation.  
- Try advanced transformer models (RoBERTa, DistilBERT).  
- Apply hyperparameter tuning and cross-validation.  
- Deploy model with FastAPI or Streamlit for interactive demo.  
- Bump pinned dependencies (torch, transformers) and re-verify the BERT path.
