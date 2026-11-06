"""Inspect Kineto encoder ranges and correlated GPU work; never report throughput."""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path

import ijson

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trace", type=Path)
parser.add_argument("--model", choices=["sensenova", "qwen"], required=True)
parser.add_argument("--mode", choices=["off", "on"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
events = []
with gzip.open(args.trace, "rb") as stream:
    for event in ijson.items(stream, "traceEvents.item", use_float=True):
        if event.get("ph") == "X" and event.get("cat") in ("cpu_op", "user_annotation", "kernel", "gpu_memcpy"):
            event["args"] = {
                key: value
                for key, value in event.get("args", {}).items()
                if key in ("External id", "bytes")
            }
            events.append(event)
steps = sorted(
    (
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event["name"].startswith("ProfilerStep#")
    ),
    key=lambda event: event["ts"],
)
if len(steps) != 15:
    raise ValueError(f"Expected 15 captured steps, found {len(steps)}")
step_starts = [step["ts"] for step in steps]
names = (
    ["SenseNovaU1ImageUnit_EditImageEmbedder", "SenseNovaU1ImageUnit_PromptEmbedder"]
    if args.model == "sensenova"
    else [
        "QwenImageEditPromptEncoder",
        "QwenImageEditReferenceVAE",
        "QwenImageEditTargetVAE",
    ]
)
ranges = [
    event
    for event in events
    if event.get("cat") == "user_annotation" and event["name"] in names
]
cpu_by_external_id = {
    event["args"]["External id"]: event
    for event in events
    if event.get("cat") in ("cpu_op", "user_annotation") and "External id" in event.get("args", {})
}
stage_ranges = collections.defaultdict(list)
for event in events:
    name = event["name"]
    if name in names and event.get("cat") == "user_annotation":
        stage = name
    elif name.startswith("autograd::engine::evaluate_function"):
        stage = "backward"
    elif name == "Optimizer.step#AdamW.step":
        stage = "optimizer"
    else:
        continue
    stage_ranges[(event["pid"], event["tid"])].append(
        (event["ts"], event["ts"] + event["dur"], stage)
    )
for intervals in stage_ranges.values():
    intervals.sort()


def step_index(timestamp):
    index = bisect.bisect_right(step_starts, timestamp) - 1
    return (
        index
        if index >= 0 and timestamp <= steps[index]["ts"] + steps[index]["dur"]
        else None
    )


def stage_for(operator):
    if operator is None:
        return "unattributed"
    intervals = stage_ranges[(operator["pid"], operator["tid"])]
    index = bisect.bisect_right(intervals, (operator["ts"], float("inf"), "")) - 1
    if index >= 0 and operator["ts"] <= intervals[index][1]:
        return intervals[index][2]
    return "other_forward_and_framework"


encoder = {}
for name in names:
    selected = [event for event in ranges if event["name"] == name]
    hot = [
        event
        for event in selected
        if (index := step_index(event["ts"])) is not None and index >= 12
    ]
    encoder[name] = {
        "calls": len(selected),
        "hot_calls": len(hot),
        "cpu_range_seconds": sum(event["dur"] for event in selected) / 1e6,
        "hot_cpu_range_seconds": sum(event["dur"] for event in hot) / 1e6,
    }
    expected_hot = 6 if args.mode == "off" or name == "QwenImageEditTargetVAE" else 0
    if not selected or len(hot) != expected_hot:
        raise ValueError(
            f"Unexpected cold/hot encoder execution: {name}: {encoder[name]}"
        )

gpu = collections.defaultdict(
    lambda: {
        "kernel_calls": 0,
        "kernel_seconds_sum": 0.0,
        "h2d_calls": 0,
        "h2d_bytes": 0,
    }
)
hot_gpu = collections.defaultdict(
    lambda: {
        "kernel_calls": 0,
        "kernel_seconds_sum": 0.0,
        "h2d_calls": 0,
        "h2d_bytes": 0,
    }
)
for event in events:
    kernel = event.get("cat") == "kernel"
    h2d = event.get("cat") == "gpu_memcpy" and "HtoD" in event["name"]
    if not kernel and not h2d:
        continue
    operator = cpu_by_external_id.get(event.get("args", {}).get("External id"))
    stage = stage_for(operator)
    index = step_index(operator["ts"]) if operator is not None else None
    groups = [gpu[stage]]
    if index is not None and index >= 12:
        groups.append(hot_gpu[stage])
    for group in groups:
        if kernel:
            group["kernel_calls"] += 1
            group["kernel_seconds_sum"] += event["dur"] / 1e6
        if h2d:
            group["h2d_calls"] += 1
            group["h2d_bytes"] += event["args"]["bytes"]
if not sum(group["kernel_calls"] for group in gpu.values()) or not sum(
    group["h2d_calls"] for group in gpu.values()
):
    raise ValueError("Trace is missing CUDA kernels or host-to-device copies")
if any(not gpu[name]["kernel_calls"] for name in names):
    raise ValueError("Cannot correlate GPU work with every recorded encoder range")
report = {
    "trace": str(args.trace),
    "model": args.model,
    "mode": args.mode,
    "scope": "Rank 0 only, 15 profiled steps / 30 samples on this rank; hot window is profiled steps 13-15 / 6 rank-local samples. CPU named-range wall times include dispatch and possible waiting, not pure GPU latency. GPU kernel durations are summed across streams and may overlap; never add them to infer end-to-end latency. GPU work is attributed by Kineto External id to its CPU operator and enclosing named range; unmatched events are reported explicitly. Other forward/framework work includes DiT and distributed/runtime overhead, not isolated DiT compute.",
    "encoder_ranges": encoder,
    "gpu_by_stage": dict(gpu),
    "hot_gpu_by_stage": dict(hot_gpu),
    "passed": True,
}
args.output.write_text(json.dumps(report, indent=2))
print(
    json.dumps({"model": args.model, "mode": args.mode, "encoder_ranges": encoder}),
    flush=True,
)
