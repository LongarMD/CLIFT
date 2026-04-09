"""Tests for JSONL I/O and small common helpers."""

import tempfile
from pathlib import Path

from clift.common import CLIFTInstance, export_jsonl, load_jsonl
from clift.tasks.functional_mappings import mod_inverse


def test_export_load_jsonl_roundtrip_utf8() -> None:
    inst = CLIFTInstance(
        task="lookup_table",
        format="demonstration",
        application="forward",
        prompt="hello — world",
        target="réponse",
        difficulty=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        export_jsonl([inst], path)
        rows = load_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["prompt"] == "hello — world"
    assert rows[0]["target"] == "réponse"


def test_mod_inverse_prime_field() -> None:
    p = 7
    a = 3
    assert (a * mod_inverse(a, p)) % p == 1
