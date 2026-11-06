"""Run one production Ray experiment and retain submission, driver and timing evidence."""

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import psutil
import ray
from ray import serve
from ray.job_submission import JobSubmissionClient


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["sensenova", "qwen"], required=True)
parser.add_argument("--mode", choices=["on", "off"], required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--dataset", type=int, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("extra", nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.extra[:1] == ["--"]:
    args.extra = args.extra[1:]
output = Path("/tmp/relax-task17/runs") / args.label
output.mkdir(exist_ok=False)
client = JobSubmissionClient("http://127.0.0.1:8265")
ray.init(address="auto", log_to_driver=False)
serve.shutdown()
ray.shutdown()
script = Path(__file__).with_name("run-ray.sh")
command = [
    "bash",
    str(script),
    args.model,
    str(2 * 1024**3 if args.mode == "on" else 0),
    str(args.steps),
    str(args.dataset),
    args.label,
    *args.extra,
]
gpu_columns = ("index", "utilization.gpu", "memory.used", "power.draw", "temperature.gpu", "clocks.current.sm")


def resources():
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        info = process.info
        command_line = " ".join(info["cmdline"] or [])
        if info["memory_info"] is not None and ("ray::" in command_line or "/relax-task17/ray-venv/" in command_line):
            processes.append(
                {
                    "pid": info["pid"],
                    "name": info["name"],
                    "rss": info["memory_info"].rss,
                }
            )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_columns)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    return {
        "monotonic": time.monotonic(),
        "processes": processes,
        "gpu_csv": gpu,
        "container_memory_bytes": int(usage.read_text()),
        "shared_memory_used_bytes": shutil.disk_usage("/dev/shm").used,
        "disk_used_bytes": shutil.disk_usage("/tmp/relax-task17").used,
    }


samples = (output / "resources.jsonl").open("w")
start = time.monotonic()
submitted = subprocess.run(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120, check=False
)
(output / "submission.log").write_text(submitted.stdout)
submitted.check_returncode()
plain = re.sub(r"\x1b\[[0-9;]*m", "", submitted.stdout)
match = re.search(r"Job '([^']+)' submitted successfully", plain)
if match is None:
    raise RuntimeError("No Ray submission ID in submission.log")
job_id = match[1]
print(json.dumps({"job_id": job_id, "label": args.label}), flush=True)
while True:
    samples.write(json.dumps(resources()) + "\n")
    samples.flush()
    status = client.get_job_status(job_id)
    if status.is_terminal():
        break
    if time.monotonic() - start > 1800:
        client.stop_job(job_id)
        raise TimeoutError(f"Our validation job {job_id} exceeded 30 minutes")
    time.sleep(2)
seconds = time.monotonic() - start
samples.close()
logs = client.get_job_logs(job_id)
(output / "driver.log").write_text(logs)
info = client.get_job_info(job_id)
result = {
    "job_id": job_id,
    "model": args.model,
    "mode": args.mode,
    "steps": args.steps,
    "dataset": args.dataset,
    "job_seconds": seconds,
    "status": str(status),
    "message": info.message,
    "ray_start_time_ms": info.start_time,
    "ray_end_time_ms": info.end_time,
    "gpu_csv_columns": gpu_columns,
    "command": command,
    "log": str(output / "driver.log"),
}
(output / "job.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
if str(status) != "SUCCEEDED":
    raise SystemExit("Ray validation failed; inspect driver.log")
