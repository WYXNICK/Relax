"""Verify the SenseNova BF16 export against a real CPU inference model."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from relax.backends.fsdp.checkpoint import _load_dcp_model_state
from relax.backends.fsdp.lora import inject_lora_adapter
from relax.models.sensenova.pipeline import SenseNovaU1ImagePipeline


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("export", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
contract = json.loads((args.checkpoint / "adapter_contract.json").read_text())
pipeline = SenseNovaU1ImagePipeline.from_pretrained(
    model_path=contract["model_path"],
    torch_dtype=torch.bfloat16,
    device=torch.device("cpu"),
)
pipeline.freeze_except([])
lora = contract["lora"]
inject_lora_adapter(
    pipeline.dit,
    rank=lora["rank"],
    alpha=lora["alpha"],
    target_modules=lora["target_modules"],
)
exported = load_file(str(args.export))
loaded = {name: parameter for name, parameter in pipeline.dit.named_parameters() if parameter.requires_grad}
source = _load_dcp_model_state(str(args.checkpoint / "fsdp"))
expected = {name.removeprefix("pipe.dit."): value.to(torch.bfloat16) for name, value in source.items()}
if expected.keys() != exported.keys() or expected.keys() != loaded.keys():
    raise ValueError("Exported adapter keyset differs from the real SenseNova model")
incompatible = pipeline.dit.load_state_dict(exported, strict=False)
if incompatible.unexpected_keys:
    raise ValueError(incompatible.unexpected_keys)
differences = [
    name
    for name in expected
    if not torch.equal(expected[name], exported[name]) or not torch.equal(expected[name], loaded[name])
]
report = {
    "checkpoint": str(args.checkpoint),
    "export": str(args.export),
    "loaded_tensors": len(loaded),
    "differences": differences,
    "passed": not differences,
}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["passed"] else 1)
