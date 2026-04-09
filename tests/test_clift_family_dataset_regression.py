"""Locked dataset regression: each CLIFT task family has a manifest + baseline JSONL."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from clift.common import FAMILIES, load_jsonl
from clift.data import _TASK_GENERATORS, generate_clift_dataset


def _fixture_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "clift_families"


def _expected_family_slugs() -> List[str]:
    return sorted(FAMILIES)


def _canonical_record(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(records: List[Dict[str, Any]]) -> str:
    blob = "\n".join(_canonical_record(record) for record in records).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_roundtrip(record: Dict[str, Any]) -> Dict[str, Any]:
    """Match JSONL round-trip so nested dict keys align with load_jsonl (str keys)."""
    return json.loads(json.dumps(record))


def _missing_tasks(manifest: Dict[str, Any]) -> List[str]:
    tasks = manifest["kwargs"]["tasks"]
    return [t for t in tasks if t not in _TASK_GENERATORS]


def _family_paths(family_slug: str) -> tuple[Path, Path, Path]:
    root = _fixture_root()
    d = root / family_slug
    return d, d / "manifest.json", d / "baseline.jsonl"


@pytest.mark.parametrize("family_slug", _expected_family_slugs())
def test_fixture_family_lists_registered_tasks(family_slug: str) -> None:
    family_dir, manifest_path, baseline_path = _family_paths(family_slug)
    assert family_dir.is_dir(), (
        f"Missing fixture directory for family {family_slug!r}: {family_dir}. "
        "Run scripts/regen_locked_datasets.py (see script docstring)."
    )
    assert manifest_path.is_file(), f"Missing manifest for family {family_slug!r}: {manifest_path}"
    assert baseline_path.is_file(), f"Missing baseline JSONL for family {family_slug!r}: {baseline_path}"
    family_manifest = json.loads(manifest_path.read_text())
    missing = _missing_tasks(family_manifest)
    assert not missing, (
        f"Fixture {family_slug} lists tasks not registered in this environment: {missing}. "
        "Install optional deps (e.g. dm-clrs) or regenerate baselines."
    )


@pytest.mark.parametrize("family_slug", _expected_family_slugs())
def test_baseline_hash_matches_manifest(family_slug: str) -> None:
    family_dir, manifest_path, baseline_path = _family_paths(family_slug)
    assert family_dir.is_dir(), f"Missing fixture directory for family {family_slug!r}: {family_dir}"
    assert baseline_path.is_file(), f"Missing baseline JSONL for family {family_slug!r}: {baseline_path}"
    family_manifest = json.loads(manifest_path.read_text())
    missing = _missing_tasks(family_manifest)
    if missing:
        pytest.skip(f"family={family_slug}: missing tasks {missing}")
    baseline_records = load_jsonl(baseline_path)
    assert _canonical_digest(baseline_records) == family_manifest["content_sha256"], (
        f"Baseline fixture hash does not match manifest lock ({family_slug})."
    )


@pytest.mark.parametrize("family_slug", _expected_family_slugs())
def test_generated_dataset_matches_locked_baseline(family_slug: str) -> None:
    family_dir, manifest_path, baseline_path = _family_paths(family_slug)
    assert family_dir.is_dir(), f"Missing fixture directory for family {family_slug!r}: {family_dir}"
    assert baseline_path.is_file(), f"Missing baseline JSONL for family {family_slug!r}: {baseline_path}"
    family_manifest = json.loads(manifest_path.read_text())
    missing = _missing_tasks(family_manifest)
    if missing:
        pytest.skip(f"family={family_slug}: missing tasks {missing}")

    baseline_records = load_jsonl(baseline_path)
    fresh_records = [
        _json_roundtrip(instance.to_dict())
        for instance in generate_clift_dataset(**family_manifest["kwargs"])
    ]

    expected_count = int(family_manifest["expected_record_count"])
    assert len(fresh_records) == expected_count
    assert len(baseline_records) == expected_count

    if baseline_records == fresh_records:
        return

    mismatch_idx = min(len(baseline_records), len(fresh_records))
    for idx, (baseline_row, fresh_row) in enumerate(zip(baseline_records, fresh_records)):
        if baseline_row != fresh_row:
            mismatch_idx = idx
            break

    if mismatch_idx >= len(baseline_records) or mismatch_idx >= len(fresh_records):
        pytest.fail(
            f"[{family_slug}] baseline mismatch: differing lengths "
            f"(baseline={len(baseline_records)}, fresh={len(fresh_records)})."
        )

    baseline_row = baseline_records[mismatch_idx]
    fresh_row = fresh_records[mismatch_idx]
    baseline_json = _canonical_record(baseline_row).splitlines()
    fresh_json = _canonical_record(fresh_row).splitlines()
    row_diff = "\n".join(
        difflib.unified_diff(
            baseline_json,
            fresh_json,
            fromfile="baseline",
            tofile="fresh",
            lineterm="",
        )
    )
    instance_id = fresh_row.get("instance_id", mismatch_idx)
    axes = (
        fresh_row.get("task"),
        fresh_row.get("format"),
        fresh_row.get("application"),
        fresh_row.get("difficulty"),
    )
    pytest.fail(
        f"[{family_slug}] baseline drift at index={mismatch_idx}, "
        f"instance_id={instance_id}, axes={axes}.\n{row_diff}"
    )
