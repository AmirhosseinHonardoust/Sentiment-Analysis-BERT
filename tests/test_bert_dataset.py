import pandas as pd
import pytest


def _tiny_df():
    return pd.DataFrame(
        {
            "text": ["good great awesome", "bad terrible awful"],
            "label": ["positive", "negative"],
        }
    )


def test_bert_tweet_dataset_shapes_and_labels(tiny_bert_dir: str):
    from transformers import AutoTokenizer

    from bert_dataset import BertTweetDataset

    tok = AutoTokenizer.from_pretrained(tiny_bert_dir)
    ds = BertTweetDataset(_tiny_df(), tok, max_len=8)

    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"].shape == (8,)
    assert item["labels"].item() == 2  # "positive" -> LABEL2ID["positive"]


def test_bert_tweet_dataset_rejects_unknown_label(tiny_bert_dir: str):
    from transformers import AutoTokenizer

    from bert_dataset import BertTweetDataset

    tok = AutoTokenizer.from_pretrained(tiny_bert_dir)
    bad_df = pd.DataFrame({"text": ["good"], "label": ["not_a_real_label"]})
    with pytest.raises(ValueError, match="not_a_real_label"):
        BertTweetDataset(bad_df, tok, max_len=8)
