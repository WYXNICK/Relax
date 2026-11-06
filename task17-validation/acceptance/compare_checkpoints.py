"""Compare trusted acceptance checkpoints, including optimizer and RNG state."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("actual", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = {
    "reference": str(args.reference),
    "actual": str(args.actual),
    "tensors": 0,
    "differences": [],
}


def compare(a, b, name):
    if isinstance(a, torch.Tensor):
        report["tensors"] += 1
        equal = isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        equal = isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        equal = isinstance(b, dict) and a.keys() == b.keys()
        if equal:
            for key in a:
                compare(a[key], b[key], f"{name}.{key}")
            return
    elif isinstance(a, (list, tuple)):
        equal = type(a) is type(b) and len(a) == len(b)
        if equal:
            for index, (left, right) in enumerate(zip(a, b, strict=True)):
                compare(left, right, f"{name}[{index}]")
            return
    else:
        equal = type(a) is type(b) and a == b
    if not equal:
        report["differences"].append(name)


for directory in (args.reference, args.actual):
    if not (directory / "COMMITTED").is_file():
        raise ValueError(f"Uncommitted checkpoint: {directory}")
with tempfile.TemporaryDirectory(prefix="task17-compare-") as temporary:
    states = []
    for index, directory in enumerate((args.reference, args.actual)):
        path = Path(temporary) / f"{index}.pt"
        dcp_to_torch_save(directory / "fsdp", path)
        states.append(torch.load(path, map_location="cpu", weights_only=False))
    compare(*states, "dcp")
for filename in ("adapter_contract.json", "trainer_state.json"):
    compare(
        *(json.loads((directory / filename).read_text()) for directory in (args.reference, args.actual)),
        filename,
    )
reference_rng = sorted(args.reference.glob("rng_rank*.pt"))
if [p.name for p in reference_rng] != sorted(p.name for p in args.actual.glob("rng_rank*.pt")):
    raise ValueError("Checkpoint rank sets differ")
for path in reference_rng:
    compare(
        torch.load(path, map_location="cpu", weights_only=False),
        torch.load(args.actual / path.name, map_location="cpu", weights_only=False),
        path.name,
    )
report["passed"] = bool(report["tensors"]) and not report["differences"]
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["passed"] else 1)
