"""Preserve production CPU preparation metrics alongside raw TensorBoard events."""

import argparse
import json
import shutil
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", type=Path)
parser.add_argument("results", type=Path)
args = parser.parse_args()
reports = []
for directory in sorted(args.runs.glob("perf-*")):
    source = directory / "checkpoints/tensorboard_log"
    destination = args.results / directory.name / "tensorboard_log"
    shutil.copytree(source, destination, dirs_exist_ok=True)
    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0}).Reload()
    records = {
        tag: [
            {"step": event.step, "value": event.value, "wall_time": event.wall_time}
            for event in accumulator.Scalars(tag)
        ]
        for tag in accumulator.Tags()["scalars"]
    }
    (destination / "scalars.json").write_text(json.dumps(records, indent=2))
    preparation = records["perf/data_wait_seconds"]
    if [row["step"] for row in preparation] != list(range(1, 33)):
        raise ValueError(f"Incomplete preparation metrics: {directory.name}")
    warmup = 12 if "-high-" in directory.name else 4
    reports.append(
        {
            "run": directory.name,
            "preparation_full_seconds": sum(row["value"] for row in preparation),
            "preparation_steady_seconds": sum(
                row["value"] for row in preparation[warmup:]
            ),
            "preparation_steady_seconds_per_step": sum(
                row["value"] for row in preparation[warmup:]
            )
            / (32 - warmup),
        }
    )
(args.results / "preparation-summary.json").write_text(
    json.dumps(
        {
            "scope": "Per-step maximum across ranks of accumulated synchronous image decode/resize/crop and condition-key lookup CPU wall time. This excludes the TQ read, rank agreement and GPU encoding. It is a stage diagnostic, not the throughput denominator; end-to-end timing includes every stage.",
            "runs": reports,
        },
        indent=2,
    )
)
