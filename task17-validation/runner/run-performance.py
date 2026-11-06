"""Execute the predeclared serial paired schedule through the production entrypoint."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
protocol = json.loads((root / "performance-protocol.json").read_text())
for model in protocol["models"]:
    for regime, settings in protocol["regimes"].items():
        runs = []
        for repeat, order in enumerate(protocol["order"], 1):
            for mode in order:
                label = f"perf-{model}-{regime}-{repeat}-{mode}"
                command = [
                    sys.executable,
                    str(root / "experiment.py"),
                    "--model",
                    model,
                    "--mode",
                    mode,
                    "--steps",
                    str(protocol["steps"]),
                    "--dataset",
                    str(settings["dataset"]),
                    "--label",
                    label,
                ]
                directory = Path("/tmp/relax-task17/runs") / label
                if not (directory / "job.json").exists():
                    print(json.dumps({"starting": label}), flush=True)
                    with (Path("/tmp/relax-task17") / f"{label}-experiment.log").open("w") as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((directory / "job.json").read_text())
                expected = {
                    "model": model,
                    "mode": mode,
                    "steps": protocol["steps"],
                    "dataset": settings["dataset"],
                    "status": "SUCCEEDED",
                }
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Existing run does not satisfy the protocol: {label}")
                destination = root.parent / "results" / label
                destination.mkdir(exist_ok=True)
                for name in (
                    "job.json",
                    "driver.log",
                    "resources.jsonl",
                    "submission.log",
                ):
                    shutil.copyfile(directory / name, destination / name)
                runs.append(
                    {
                        "mode": mode,
                        "log": f"{label}/driver.log",
                        "job_seconds": result["job_seconds"],
                    }
                )
                print(
                    json.dumps({"completed": label, "seconds": result["job_seconds"]}),
                    flush=True,
                )
        manifest = {
            "world_size": protocol["world_size"],
            "warmup_steps": settings["warmup_steps"],
            "runs": runs,
        }
        manifest_path = root.parent / "results" / f"{model}-{regime}-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        subprocess.run(
            [
                sys.executable,
                str(root.parent / "acceptance/summarize_frozen_conditions.py"),
                str(manifest_path),
                "--output",
                str(manifest_path.with_name(f"{model}-{regime}-summary.json")),
            ],
            check=True,
        )
