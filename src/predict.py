"""Run sentiment prediction on one or more texts using a trained BERT or LSTM checkpoint.

Auto-detects the checkpoint type the same way evaluate.py does. Prints the
predicted label and per-class probabilities for each input text.
"""

import argparse
from typing import List, Tuple

import torch

from evaluate import find_lstm_checkpoint, is_hf_dir, is_lstm_checkpoint
from utils import ID2LABEL, encode_lstm_text


def predict_bert(
    texts: List[str], ckpt_dir: str, max_len: int = 128
) -> List[Tuple[str, List[float]]]:
    """Classify `texts` with an HF-format BERT checkpoint directory."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir).to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tok(
                text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            results.append((ID2LABEL.get(pred, str(pred)), probs.tolist()))
    return results


def predict_lstm(texts: List[str], ckpt_path: str) -> List[Tuple[str, List[float]]]:
    """Classify `texts` with an LSTM checkpoint (file or its containing directory)."""
    from train_lstm import LSTMClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_lstm_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]
    embed_dim = ckpt.get("embed_dim", 100)
    hidden_dim = ckpt.get("hidden_dim", 128)

    model = LSTMClassifier(len(vocab), embed_dim, hidden_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            ids = encode_lstm_text(vocab, text, max_len)
            xb = torch.tensor([ids], dtype=torch.long).to(device)
            probs = torch.softmax(model(xb), dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            results.append((ID2LABEL.get(pred, str(pred)), probs.tolist()))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "For BERT: directory with model files (config.json, tokenizer.json). "
            "For LSTM: the best_model_lstm.pt file, or its containing directory."
        ),
    )
    ap.add_argument(
        "--text",
        action="append",
        required=True,
        help="Text to classify. Repeat --text for multiple inputs.",
    )
    ap.add_argument(
        "--max-len", type=int, default=128, help="Max token sequence length (BERT path only)."
    )
    args = ap.parse_args()

    if is_hf_dir(args.checkpoint):
        results = predict_bert(args.text, args.checkpoint, max_len=args.max_len)
    elif is_lstm_checkpoint(args.checkpoint):
        results = predict_lstm(args.text, args.checkpoint)
    else:
        raise ValueError(
            "--checkpoint must be either a BERT model directory (containing "
            "config.json) or an LSTM checkpoint (best_model_lstm.pt / its directory)."
        )

    for text, (label, probs) in zip(args.text, results, strict=True):
        prob_str = ", ".join(f"{ID2LABEL.get(i, str(i))}={p:.3f}" for i, p in enumerate(probs))
        print(f"[{label}] ({prob_str})  {text!r}")


if __name__ == "__main__":
    main()
