import pandas as pd
import torch

from train_lstm import LSTMClassifier, TextDS


def _tiny_df():
    return pd.DataFrame(
        {
            "text": ["good great awesome", "bad terrible awful", "ok fine neutral thing"],
            "label": ["positive", "negative", "neutral"],
        }
    )


def test_textds_builds_vocab_with_pad_and_unk():
    ds = TextDS(_tiny_df(), build=True, max_len=8)
    assert ds.vocab["<pad>"] == 0
    assert ds.vocab["<unk>"] == 1
    assert "good" in ds.vocab


def test_textds_encode_pads_short_sequences():
    ds = TextDS(_tiny_df(), build=True, max_len=8)
    ids = ds.encode("good great")
    assert len(ids) == 8
    assert ids[-1] == 0  # padded with <pad> id


def test_textds_encode_truncates_long_sequences():
    ds = TextDS(_tiny_df(), build=True, max_len=2)
    ids = ds.encode("good great awesome")
    assert len(ids) == 2


def test_textds_encode_maps_unseen_words_to_unk():
    ds = TextDS(_tiny_df(), build=True, max_len=8)
    ids = ds.encode("totallyunseenword")
    assert ids[0] == 1  # <unk>


def test_textds_reuses_fixed_vocab_for_val_set():
    train_ds = TextDS(_tiny_df(), build=True, max_len=8)
    val_ds = TextDS(_tiny_df(), vocab=train_ds.vocab, max_len=8)
    assert val_ds.vocab is train_ds.vocab


def test_lstm_classifier_forward_shape():
    model = LSTMClassifier(vocab_size=10, embed_dim=4, hidden_dim=6, num_classes=3)
    x = torch.randint(0, 10, (2, 5))  # batch of 2, seq len 5
    out = model(x)
    assert out.shape == (2, 3)
