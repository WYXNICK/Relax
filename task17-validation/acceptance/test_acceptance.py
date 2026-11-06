# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Acceptance must not hide slow ranks, cold cost, missing logs or zero
norms."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from summarize_frozen_conditions import interval_seconds, read_steps, summarize
from validate_frozen_conditions import compare_tensors, passes


def test_summary_counts_slowest_rank_cold_and_job_cost(tmp_path):
    runs = []
    for mode, durations in (("off", [2, 2, 2]), ("on", [4, 1, 1])):
        lines = []
        finished = {0: 1000.0, 1: 2000.0}
        for step, seconds in enumerate(durations, 1):
            for rank in range(2):
                duration = seconds if rank else seconds / 2
                finished[rank] += duration
                row = {
                    "step": step,
                    "rank": rank,
                    "samples": 2,
                    "seconds": duration if step == 1 else duration - 0.25,
                    "finished_at": finished[rank],
                    "cache": None,
                    "peak_allocated_bytes": 128,
                    "peak_reserved_bytes": 256,
                }
                lines.append("actor: i2i SFT measurement " + json.dumps(row))
        path = tmp_path / f"{mode}.log"
        path.write_text("\n".join(lines))
        runs.extend(
            {"mode": mode, "log": path.name, "job_seconds": 10} for _ in range(3)
        )
    report = summarize({"world_size": 2, "warmup_steps": 1, "runs": runs}, tmp_path)
    assert report["steady_speedup_percent"] == 100
    assert report["job_speedup_percent"] == 0
    assert report["estimated_extra_steady_steps_to_repay_cold_overhead"] == 2
    assert report["medians"]["on"]["training_seconds"] == 6
    assert report["first_sustained_training_break_even_step"] == 3
    assert report["first_sustained_training_break_even_samples"] == 6
    with pytest.raises(ValueError, match="Incomplete ranks"):
        read_steps(tmp_path / "on.log", 3)


def test_acceptance_rejects_nonzero_error_against_zero_reference_and_nonfinite():
    assert not passes(
        {"x": {"finite": True, "exact": False, "relative_l2": None}}, exact=False
    )
    assert not passes(
        {"x": {"finite": False, "exact": True, "relative_l2": 0}}, exact=False
    )
    assert not passes(
        {"x": {"finite": True, "exact": False, "relative_l2": 0.001}}, exact=True
    )
    assert passes({"x": {"finite": True, "exact": True, "relative_l2": 0}}, exact=True)
    assert not passes(
        {"x": {"finite": True, "exact": False, "relative_l2": 0.001, "max_abs": 1}},
        exact=False,
        atol=1e-6,
    )


def test_interval_does_not_double_count_cross_step_rank_waits():
    steps = [
        {"seconds_by_rank": {0: 11.0, 1: 1.0}},
        {"seconds_by_rank": {0: 1.0, 1: 11.0}},
    ]
    assert interval_seconds(steps, 2) == 12


def _compare_shards(rank, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        reference = {
            "weight": torch.tensor([3.0 if rank == 0 else 4.0]),
            "zero": torch.zeros(1),
        }
        actual = {
            "weight": reference["weight"] + (0.5 if rank == 0 else 0),
            "zero": torch.zeros(1),
        }
        result = compare_tensors(reference, actual, dist.group.WORLD)
        if rank == 0:
            Path(output).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_error_norms_are_reduced_across_shards_before_division(tmp_path):
    output = tmp_path / "comparison.json"
    mp.spawn(
        _compare_shards,
        args=(str(tmp_path / "rendezvous"), str(output)),
        nprocs=2,
        join=True,
    )
    result = json.loads(output.read_text())
    assert result["weight"]["reference_l2"] == 5
    assert result["weight"]["relative_l2"] == pytest.approx(0.1)
    assert result["weight"]["max_abs"] == 0.5
    assert result["zero"]["relative_l2"] == 0
