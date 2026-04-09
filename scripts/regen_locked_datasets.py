"""
Regenerate `data/clift.jsonl` and locked family baselines from their manifests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from clift.common import export_jsonl, load_jsonl  # noqa: E402
from clift.data import generate_clift_dataset  # noqa: E402


def _digest_records(records: list) -> tuple[str, int]:
    def canon(r: dict) -> str:
        return json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    blob = "\n".join(canon(r) for r in records).encode("utf-8")
    return hashlib.sha256(blob).hexdigest(), len(records)


def regen_locked_jsonl(manifest_rel: str, output_rel: str) -> None:
    manifest_path = ROOT / manifest_rel
    out_path = ROOT / output_rel
    manifest = json.loads(manifest_path.read_text())
    gen_kw = {**manifest["kwargs"], "show_progress": True}
    instances = generate_clift_dataset(**gen_kw)
    export_jsonl(instances, out_path)
    records = load_jsonl(out_path)
    digest, n = _digest_records(records)
    manifest["expected_record_count"] = n
    manifest["content_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_rel, "records", n, "sha256", digest)


def main() -> None:
    regen_locked_jsonl("data/manifest.json", "data/clift.jsonl")



if __name__ == "__main__":
    main()
