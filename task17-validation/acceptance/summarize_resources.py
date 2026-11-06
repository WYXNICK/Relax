"""Summarize resource samples and startup latency from production experiments."""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from summarize_frozen_conditions import read_steps


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reports = []
for directory in sorted(args.results.glob("perf-*")):
    job = json.loads((directory / "job.json").read_text())
    samples = [
        json.loads(line)
        for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    steps = read_steps(directory / "driver.log", 2)
    first_train = re.search(
        r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Actor training step \d+/",
        (directory / "driver.log").read_text(),
    )
    if first_train is None:
        raise ValueError(f"Missing first training timestamp: {directory}")
    start = (
        datetime.strptime(first_train[1], "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    gpu_rows = [
        dict(zip(job["gpu_csv_columns"], (value.strip() for value in row), strict=True))
        for sample in samples
        for row in csv.reader(sample["gpu_csv"].splitlines())
    ]
    report = {
        "run": directory.name,
        "ray_start_to_first_training_seconds": start - job["ray_start_time_ms"] / 1000,
        "startup_timestamp_resolution_seconds": 1,
        "container_memory_peak_bytes": max(
            sample["container_memory_bytes"] for sample in samples
        ),
        "ray_process_rss_sum_peak_bytes": max(
            sum(p["rss"] for p in sample["processes"]) for sample in samples
        ),
        "shared_memory_start_bytes": samples[0]["shared_memory_used_bytes"],
        "shared_memory_peak_bytes": max(
            sample["shared_memory_used_bytes"] for sample in samples
        ),
        "shared_memory_end_bytes": samples[-1]["shared_memory_used_bytes"],
        "filesystem_used_start_bytes": samples[0]["disk_used_bytes"],
        "filesystem_used_end_bytes": samples[-1]["disk_used_bytes"],
        "gpu_peak_used_mib_by_device": {
            index: max(
                float(row["memory.used"]) for row in gpu_rows if row["index"] == index
            )
            for index in sorted({row["index"] for row in gpu_rows})
        },
        "torch_peak_allocated_bytes": max(
            step["peak_allocated_bytes"] for step in steps
        ),
        "torch_peak_reserved_bytes": max(step["peak_reserved_bytes"] for step in steps),
        "cache_final_by_rank": steps[-1]["cache_by_rank"],
    }
    reports.append(report)
args.output.write_text(
    json.dumps(
        {
            "scope": "Two-second samples over the entire job, including CPU startup and teardown. RSS sums include shared pages more than once; container charged memory includes file cache. Filesystem usage includes intentionally retained checkpoints/logs, not only cache. Cache owns no disk or shared-memory files. GPU peaks are per device, not summed.",
            "runs": reports,
        },
        indent=2,
    )
)
