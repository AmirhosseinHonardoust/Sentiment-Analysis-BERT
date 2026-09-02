import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def tiny_bert_dir(tmp_path):
    """A tiny, randomly-initialized local BERT checkpoint (config + tokenizer + weights).

    Lets tests exercise the real train_bert.py / evaluate.py / predict.py BERT code
    paths via AutoTokenizer/AutoModelForSequenceClassification.from_pretrained
    against a local directory — no network access and no real pretrained weights
    required. This is what lets the BERT path run in CI at all; previously it was
    entirely untested there because it needed a Hugging Face Hub download.
    """
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    outdir = tmp_path / "tiny_bert_base"
    outdir.mkdir()

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        "good",
        "great",
        "awesome",
        "excellent",
        "bad",
        "terrible",
        "awful",
        "disappointing",
        "ok",
        "fine",
        "average",
        "normal",
        "example",
    ]
    vocab_path = outdir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab))

    tok = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    tok.save_pretrained(str(outdir))

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=3,
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(str(outdir))

    return str(outdir)
