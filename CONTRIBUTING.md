# Contributing

## Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

pip install -r requirements-dev.txt
```

## Before opening a PR
Run the same checks CI runs, in this order:
```bash
ruff check --select E,F,I,B,SIM,UP --line-length 100 src/ tests/
black --check --line-length 100 src/ tests/
mypy --ignore-missing-imports src/
pytest
```
All four must pass. `black` (without `--check`) and `ruff ... --fix` will
auto-fix most formatting/import issues for you.

## What's covered by tests, and how
- **Preprocessing, LSTM training/eval, and utils** run for real against tiny
  in-memory datasets — no network access needed.
- **BERT training/eval/predict** (`tests/test_train_bert.py`,
  `tests/test_bert_dataset.py`) run for real too, but against a tiny,
  randomly-initialized local BERT checkpoint built by the `tiny_bert_dir`
  fixture in `tests/conftest.py` — not a real download of `bert-base-uncased`.
  This exercises the actual code paths (tokenization, training loop,
  checkpoint save/load, evaluation, inference) without needing Hugging Face
  Hub access, so it runs in CI on every push.
- **A real fine-tuning run** (`python src/train_bert.py --model
  bert-base-uncased ...`) still requires a live Hugging Face Hub download and
  isn't exercised in CI — only manually, in an environment with network
  access.

## Conventions
- Keep CLI flags documented with `help=` text — see any `src/*.py` `main()`
  for the pattern.
- Prefer extending `utils.py` / `bert_dataset.py` over duplicating
  encoding/dataset logic between train and eval scripts.
- Map any raw CSV label through `utils.label_ids()` rather than indexing
  `LABEL2ID` directly, so unexpected label values fail with a clear error.
- Dependency bumps: prefer verifying against the `tiny_bert_dir`-based BERT
  tests and the LSTM end-to-end test before merging, since those are the
  closest thing this repo has to integration coverage. Dependabot
  (`.github/dependabot.yml`) opens weekly PRs for pip and GitHub Actions
  updates; each one should still pass the full gate before merging.
