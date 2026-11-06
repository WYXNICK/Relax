"""Load the real exported adapter through the public Diffusers API on CPU."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import QwenImageEditPipeline

from relax.backends.fsdp.checkpoint import _load_dcp_model_state


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("export", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
contract = json.loads((args.checkpoint / "adapter_contract.json").read_text())
pipeline = QwenImageEditPipeline.from_pretrained(
    contract["model_path"], torch_dtype=torch.bfloat16, local_files_only=True
)
pipeline.load_lora_weights(str(args.export), weight_name="pytorch_lora_weights.safetensors", adapter_name="acceptance")
source = _load_dcp_model_state(str(args.checkpoint / "fsdp"))
loaded = {name: parameter for name, parameter in pipeline.transformer.named_parameters() if ".lora_" in name}
expected = {
    name.removeprefix("transformer.").replace(".default.", ".acceptance."): value for name, value in source.items()
}
if expected.keys() != loaded.keys():
    raise ValueError("Exported adapter keyset differs after Diffusers loading")
differences = [name for name in expected if not torch.equal(expected[name].to(loaded[name].dtype), loaded[name])]
scaling = {
    float(module.scaling["acceptance"])
    for module in pipeline.transformer.modules()
    if hasattr(module, "scaling") and "acceptance" in module.scaling
}
report = {
    "checkpoint": str(args.checkpoint),
    "export": str(args.export),
    "loaded_tensors": len(loaded),
    "loaded_dtypes": sorted({str(parameter.dtype) for parameter in loaded.values()}),
    "differences": differences,
    "scaling": sorted(scaling),
    "passed": not differences and scaling == {contract["lora"]["scaling"]},
}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["passed"] else 1)
