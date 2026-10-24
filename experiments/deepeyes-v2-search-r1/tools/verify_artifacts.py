#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check the curated historical evidence without network or GPU access."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT.parents[1] / "scripts/training/multimodal/run-qwen36-deepeyes-search-r1.sh"


def sha256(path: Path) -> str:
    """Hash one small, curated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_manifest() -> dict:
    """Describe the export only; private input records are never modified."""
    files = {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact-manifest.json" and "__pycache__" not in path.parts
    }
    return {
        "schema_version": 1,
        "historical_base_commit": "e5e1df624e8b0ea635b6f2f8a5c41c83c33958e0",
        "scope": "Public personal study branch; curated records, no new training or evaluation",
        "files": files,
        "recipe": {"repository_path": str(RECIPE.relative_to(ROOT.parents[1])), "sha256": sha256(RECIPE)},
    }


def rows(name: str) -> list[dict]:
    """Read an exported CSV without changing its numeric values."""
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify() -> dict:
    """Verify exact hashes, counts, evidence mapping, and source image
    sizes."""
    manifest = json.loads((ROOT / "artifact-manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = ROOT / name
        assert path.stat().st_size == item["bytes"], name
        assert sha256(path) == item["sha256"], name
    assert sha256(RECIPE) == manifest["recipe"]["sha256"]

    search = [item for item in rows("data/evaluation.csv") if item["dataset"] == "search"]
    assert [int(item["completed_updates"]) for item in search] == [0, 50, 100, 150, 200]
    assert [int(item["correct"]) for item in search] == [845, 1199, 1301, 1268, 1136]
    assert all(int(item["observed_samples"]) == int(item["planned_samples"]) == 2048 for item in search)
    assert max(search, key=lambda item: float(item["accuracy_planned_denominator"]))["completed_updates"] == "100"

    train = rows("data/training-200-updates.csv")
    assert [int(item["native_step"]) for item in train] == list(range(200))
    assert all(float(item["train/global_batch_size"]) == 256 for item in train)
    fixed = rows("data/fixed-16-cases.csv")
    assert len(fixed) == 80 and len({item["source_id"] for item in fixed}) == 16
    assert all(int(item["samples"]) == 8 for item in fixed)
    windows = rows("data/observed-two-hour-sm.csv")
    assert len(windows) == 64 and all(float(item["mean_sm_percent"]) > 20 for item in windows)

    provenance = json.loads((ROOT / "evidence/export-provenance.json").read_text())
    scalar_bytes = gzip.decompress((ROOT / "data/scalars.csv.gz").read_bytes())
    assert hashlib.sha256(scalar_bytes).hexdigest() == provenance["uncompressed_scalar_csv_sha256"]
    scalars = list(csv.DictReader(io.StringIO(scalar_bytes.decode())))
    gradient_steps = {int(item["step"]) for item in scalars if item["tag"].removeprefix("./") == "train/grad_norm"}
    assert gradient_steps == set(range(200))

    image_sizes = {
        "01-training-evaluation-hd.png": (2147, 660),
        "02-performance-hd.png": (2147, 660),
        "03-training-log-steps-000-009-hd.png": (1800, 750),
    }
    for name, expected in image_sizes.items():
        header = (ROOT / "figures" / name).read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", header[16:24]) == expected
    for item in provenance["images"]:
        assert sha256(ROOT / item["file"]) == item["sha256"]

    completion = json.loads((ROOT / "evidence/evaluation-completion.json").read_text())
    assert completion["exit_code"] == 0
    assert completion["merge_receipt"]["optimizer_updates_in_evaluation"] == 0
    assert completion["merge_receipt"]["sample_counts"] == {"eval/search": 2048, "eval/vstar": 1528}
    return {
        "artifact_files": len(manifest["files"]),
        "unique_training_steps": 200,
        "search_evaluation_points": 5,
        "fixed_case_groups": 80,
        "original_images_unchanged": True,
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-manifest", action="store_true", help="Refresh the derived file list after an intentional edit"
    )
    args = parser.parse_args()
    if args.update_manifest:
        (ROOT / "artifact-manifest.json").write_text(
            json.dumps(create_manifest(), ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
