from america_ai.config import CONCEPTS
from america_ai.datasets import load_pairs, validate_pairs


def test_all_america_datasets_are_valid():
    for concept in CONCEPTS:
        pairs = load_pairs(concept)
        validate_pairs(pairs)
        assert len(pairs) == 90
        assert {pair.split for pair in pairs} == {"train", "validation", "test"}


def test_dataset_ids_are_unique_and_splits_do_not_overlap():
    pairs = load_pairs("trump_approval")
    ids = [pair.id for pair in pairs]
    assert len(ids) == len(set(ids))
    split_text = {}
    for split in ["train", "validation", "test"]:
        split_text[split] = {
            (pair.prompt, pair.positive, pair.negative)
            for pair in pairs
            if pair.split == split
        }
    assert not split_text["train"] & split_text["validation"]
    assert not split_text["train"] & split_text["test"]
