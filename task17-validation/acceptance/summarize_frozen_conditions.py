# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Summarize paired Ray logs, including cold and checkpoint steps.

Input manifest: {world_size, warmup_steps, runs: [{mode: off|on, log,
job_seconds}]}. Run each mode at least three times, separately for each model
and reuse regime. job_seconds is submission-to-completion wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def read_steps(path: Path, world: int) -> list[dict]:
    records: dict[int, dict[int, dict]] = {}
    marker = "i2i SFT measurement "
    for line in path.read_text().splitlines():
        if marker not in line:
            continue
        row, _ = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])
        ranks = records.setdefault(row["step"], {})
        if row["rank"] in ranks and ranks[row["rank"]] != row:
            raise ValueError(
                f"Conflicting measurement at step {row['step']} rank {row['rank']}"
            )
        ranks[row["rank"]] = row
    if not records or sorted(records) != list(range(1, len(records) + 1)):
        raise ValueError("Expected complete uninterrupted logs starting at step 1")
    steps = []
    previous = {}
    for step, ranks in sorted(records.items()):
        if set(ranks) != set(range(world)):
            raise ValueError(f"Incomplete ranks at step {step}")
        if len({row["samples"] for row in ranks.values()}) != 1:
            raise ValueError("Ranks disagree on global batch size")
        intervals = {
            rank: row["finished_at"] - previous[rank]
            if rank in previous
            else row["seconds"]
            for rank, row in ranks.items()
        }
        previous = {rank: row["finished_at"] for rank, row in ranks.items()}
        seconds = max(intervals.values())
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Step durations must be finite and positive")
        steps.append(
            {
                "step": step,
                "seconds": seconds,
                "seconds_by_rank": intervals,
                "samples": ranks[0]["samples"],
                "peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in ranks.values()
                ),
                "peak_reserved_bytes": max(
                    row["peak_reserved_bytes"] for row in ranks.values()
                ),
                "cache_by_rank": {rank: row["cache"] for rank, row in ranks.items()},
            }
        )
    return steps


def interval_seconds(steps: list[dict], world: int) -> float:
    # Rank boundaries are asynchronous. Sum each rank's interval first:
    # summing per-step maxima can count cross-step waiting more than once.
    return max(
        sum(row["seconds_by_rank"][rank] for row in steps) for rank in range(world)
    )


def summarize(manifest: dict, root: Path) -> dict:
    warmup, world = manifest["warmup_steps"], manifest["world_size"]
    if type(warmup) is not int or warmup < 0 or type(world) is not int or world <= 0:
        raise ValueError("Invalid world_size/warmup_steps")
    results = {"off": [], "on": []}
    trajectories = {"off": [], "on": []}
    schedules = []
    for run in manifest["runs"]:
        steps = read_steps(root / run["log"], world)
        if (
            len(steps) <= warmup
            or not math.isfinite(run["job_seconds"])
            or run["job_seconds"] <= 0
        ):
            raise ValueError("Need positive job time and steps beyond the fixed warmup")
        schedules.append([row["samples"] for row in steps])
        trajectories[run["mode"]].append(steps)
        steady = steps[warmup:]
        total_seconds = interval_seconds(steps, world)
        steady_seconds = interval_seconds(steady, world)
        if run["job_seconds"] < total_seconds:
            raise ValueError("Job wall time cannot be less than measured training time")
        results[run["mode"]].append(
            {
                "log": run["log"],
                "job_seconds": run["job_seconds"],
                "training_seconds": total_seconds,
                "cold_seconds": interval_seconds(steps[:warmup], world),
                "steady_step_seconds": steady_seconds / len(steady),
                "steady_samples_per_second": sum(row["samples"] for row in steady)
                / steady_seconds,
                "job_samples_per_second": sum(row["samples"] for row in steps)
                / run["job_seconds"],
                "peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in steps
                ),
                "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in steps),
                "final_cache_by_rank": steps[-1]["cache_by_rank"],
            }
        )
    if any(schedule != schedules[0] for schedule in schedules) or any(
        len(runs) < 3 for runs in results.values()
    ):
        raise ValueError(
            "Need at least three repeats per mode with identical sample/step schedules"
        )
    medians = {
        mode: {
            key: statistics.median(run[key] for run in runs)
            for key in (
                "job_seconds",
                "training_seconds",
                "cold_seconds",
                "steady_step_seconds",
                "steady_samples_per_second",
                "job_samples_per_second",
            )
        }
        for mode, runs in results.items()
    }
    baseline, cached = medians["off"], medians["on"]
    saving = baseline["steady_step_seconds"] - cached["steady_step_seconds"]
    cold_overhead = max(0, cached["cold_seconds"] - baseline["cold_seconds"])
    cumulative_savings = [
        statistics.median(
            interval_seconds(steps[:end], world) for steps in trajectories["off"]
        )
        - statistics.median(
            interval_seconds(steps[:end], world) for steps in trajectories["on"]
        )
        for end in range(1, len(schedules[0]) + 1)
    ]
    break_even = next(
        (
            index + 1
            for index in range(len(cumulative_savings))
            if min(cumulative_savings[index:]) >= 0
        ),
        None,
    )
    return {
        "runs": results,
        "medians": medians,
        "dispersion": {
            mode: {
                key: {
                    "mean": statistics.mean(run[key] for run in runs),
                    "stdev": statistics.stdev(run[key] for run in runs),
                    "min": min(run[key] for run in runs),
                    "max": max(run[key] for run in runs),
                }
                for key in ("steady_samples_per_second", "job_seconds")
            }
            for mode, runs in results.items()
        },
        "steady_speedup_percent": (
            cached["steady_samples_per_second"] / baseline["steady_samples_per_second"]
            - 1
        )
        * 100,
        "job_speedup_percent": (
            cached["job_samples_per_second"] / baseline["job_samples_per_second"] - 1
        )
        * 100,
        "cold_window_net_change_seconds": cached["cold_seconds"]
        - baseline["cold_seconds"],
        "median_cumulative_training_savings_seconds": cumulative_savings,
        "first_sustained_training_break_even_step": break_even,
        "first_sustained_training_break_even_samples": sum(schedules[0][:break_even])
        if break_even
        else None,
        "estimated_extra_steady_steps_to_repay_cold_overhead": math.ceil(
            cold_overhead / saving
        )
        if saving > 0
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.manifest.read_text()), args.manifest.parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
