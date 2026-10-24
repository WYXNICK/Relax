#!/usr/bin/env python3
"""Freeze this experiment's data using the repository's V2 prompt converter.

Run on the training worker; large artifacts stay on its local disk.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import unicodedata
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(row):
    question = row["extra_info"]["question"]
    question = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    images = tuple(hashlib.sha256(x["bytes"]).hexdigest() for x in row["images"])
    return question, images


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        yield from batch.to_pylist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    args = p.parse_args()
    raw = args.assets / "deepeyes/data/raw"
    out = args.assets / "deepeyes/data/experiment"
    out.mkdir(parents=True, exist_ok=True)
    source = args.repo / "examples/deepeyes_v2_agentic/convert_tool/rl_data_convert.py"
    spec = importlib.util.spec_from_file_location("official_converter", source)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    seen, representatives, groups = set(), [], {}
    for row in rows(raw / "search.parquet"):
        key = identity(row)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(row)
        groups.setdefault(key[1][0], []).append(row)
    hashes = sorted(groups)
    random.Random(1234).shuffle(hashes)
    held_images, held = set(), []
    for h in hashes:
        if len(held) + len(groups[h]) <= 256:
            held_images.add(h)
            held.extend(groups[h])
        if len(held) == 256:
            break
    assert len(held) == 256
    holdout_ids = [r["extra_info"]["index"] for r in held]
    holdout_hash = hashlib.sha256(("\n".join(holdout_ids) + "\n").encode()).hexdigest()
    assert holdout_hash == "de1e712359fb758309aa8408edd392ed2ad3bc4d00429579aff35f18742d8810"
    train = [r for r in representatives if identity(r)[1][0] not in held_images]
    vstar = list(rows(raw / "vstar_test.parquet"))
    eval_images = held_images | {h for r in vstar for h in identity(r)[1]}
    assert not {h for r in train for h in identity(r)[1]} & eval_images
    train_keys = {identity(r) for r in train}
    selections = [
        ("search_train", train, "search"),
        ("search_holdout", held, "search"),
        ("vstar_test", vstar, "vstar-test"),
    ]
    excluded = {}
    for stem, source_name, tag in [
        ("perception_train", "perception_all_1", "perception"),
        ("reason_train", "reason", "reason"),
    ]:
        selected, dropped = [], 0
        for row in rows(raw / (source_name + ".parquet")):
            key = identity(row)
            if set(key[1]) & eval_images or key in train_keys:
                dropped += 1
                continue
            train_keys.add(key)
            selected.append(row)
            if len(selected) == 1000:
                break
        assert len(selected) == 1000
        selections.append((stem, selected, tag))
        excluded[stem] = dropped
    files = {}
    for name, selected, tag in selections:
        path = out / (name + ".parquet")
        temp = path.with_suffix(".parquet.incomplete")
        with pq.ParquetWriter(
            temp,
            converter._patch_table(pa.Table.from_pylist(selected[:1]), tag, unified_schema=True).schema,
            compression="snappy",
        ) as writer:
            for start in range(0, len(selected), 64):
                table = converter._patch_table(
                    pa.Table.from_pylist(selected[start : start + 64]), tag, unified_schema=True
                )
                writer.write_table(table)
        temp.replace(path)
        files[name] = {
            "path": str(path),
            "rows": len(selected),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "ids": [r["extra_info"]["index"] for r in selected],
        }
        print(name, len(selected), flush=True)
    cache_dir = args.assets / "image-search-cache"
    archive = cache_dir / "cached_images/train.zip"
    with zipfile.ZipFile(archive) as z:
        dest = cache_dir / "cached_images"
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("Unsafe ZIP member")
        z.extractall(dest)
    cache_source = cache_dir / "fvqa_train_image_search_results_cache.json"
    cache = json.loads(cache_source.read_text())
    image_count = 0
    for item in cache.values():
        paths = []
        for name in item["cached_images_path"]:
            if not name:
                paths.append(None)
                continue
            path = cache_dir / name
            with Image.open(path) as image:
                image.verify()
            image_count += 1
            paths.append(str(path))
        item["cached_images_path"] = paths
    cache_output = cache_dir / "converted-cache.json"
    cache_output.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
    report = {
        "seed": 1234,
        "converter_sha256": sha(source),
        "search_raw_rows": 4856,
        "search_duplicate_rows_removed": 120,
        "search_holdout_ids_sha256": holdout_hash,
        "image_group_isolation": True,
        "auxiliary_excluded_before_selection": excluded,
        "selection_rule": "Search dedup question+image; shuffled image groups; auxiliary first 1000 eligible rows",
        "files": files,
        "cache": {"path": str(cache_output), "sha256": sha(cache_output), "decoded_images": image_count},
        "fixed_case_ids": holdout_ids[:16],
        "limitations": [
            "Byte-identical image isolation, not perceptual duplicate detection",
            "Wiki18 does not cover all recent events; holdout is not filtered by outcomes",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("DATA_READY", flush=True)


if __name__ == "__main__":
    main()
