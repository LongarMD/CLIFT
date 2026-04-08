"""Tests for clift.eval scoring and text helpers."""

import pytest

from clift.eval import (
    _spatial_distance_diagnostics,
    compute_soft_score,
    contains_match,
    exact_match,
    extract_list,
    first_line_match,
    normalize_text,
    score_dataset,
    single_token_compliance,
)


def test_normalize_text_strips_and_lowercases() -> None:
    assert normalize_text("  Hello, World!  ") == "hello, world"


def test_exact_match_ignores_case_and_outer_space() -> None:
    assert exact_match("  Bear  ", "bear")


def test_first_line_match() -> None:
    assert first_line_match("ant\nextra", "ant")
    assert not first_line_match("ant\nextra", "bear")


def test_extract_list_last_bracket_list() -> None:
    assert extract_list("thought [1, 2] then [3, 4]") == [3, 4]


def test_extract_list_truncated_fallback() -> None:
    assert extract_list("answer [1, 2, ") == [1, 2]


def test_contains_match_integer_uses_tail_numbers() -> None:
    assert contains_match("step 1 then 2 then 42", "42")
    assert not contains_match("step 1 then 2 then 3", "99")


def test_compute_soft_score_bounds_pair() -> None:
    assert compute_soft_score("[0, 4]", "[1, 3]") == pytest.approx(3 / 5)


def test_single_token_compliance() -> None:
    assert single_token_compliance("kitchen")
    assert single_token_compliance('"panda"')
    assert not single_token_compliance("two words")


def test_spatial_distance_diagnostics() -> None:
    inst = {
        "task": "spatial_translation",
        "latent_structure": {
            "token_map": {"a": [0, 0], "b": [2, 1]},
        },
    }
    in_vocab, dist = _spatial_distance_diagnostics(inst, "a", "b")
    assert in_vocab is True
    assert dist == 3


def test_score_dataset_length_mismatch_raises() -> None:
    instances = [{"task": "t", "format": "f", "application": "a", "target": "x"}]
    with pytest.raises(ValueError, match="same length"):
        score_dataset(instances, [])


def test_score_dataset_strict_pairing() -> None:
    instances = [
        {
            "instance_id": 0,
            "task": "lookup_table",
            "format": "demonstration",
            "application": "forward",
            "target": "cat",
            "difficulty": 1,
        },
        {
            "instance_id": 1,
            "task": "lookup_table",
            "format": "demonstration",
            "application": "forward",
            "target": "dog",
            "difficulty": 1,
        },
    ]
    preds = ["cat", "wrong"]
    df = score_dataset(instances, preds)
    assert len(df) == 2
    assert bool(df.iloc[0]["correct_extracted"])
    assert not bool(df.iloc[1]["correct_extracted"])


def test_score_dataset_masks_filtered() -> None:
    instances = [
        {
            "task": "t",
            "format": "f",
            "application": "a",
            "target": "[MASKED]",
        },
    ]
    df = score_dataset(instances, [""])
    assert len(df) == 0
