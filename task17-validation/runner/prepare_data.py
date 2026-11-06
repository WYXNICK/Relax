import hashlib
import io
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

root = Path("/tmp/relax-task17/data")
root.mkdir(parents=True, exist_ok=True)
repo = "timbrooks/instructpix2pix-clip-filtered"
info = HfApi().dataset_info(repo, revision="aa665b890915f7a42f8615bee868a9f3447e178f")
shard = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))[
    0
]
print("Downloading data", info.sha, shard, flush=True)
path = hf_hub_download(
    repo,
    shard,
    repo_type="dataset",
    revision=info.sha,
    cache_dir="/tmp/relax-task17/hf",
)
rows = next(pq.ParquetFile(path).iter_batches(batch_size=128)).to_pylist()
metadata = []
for i, row in enumerate(rows):
    for field, label in [("original_image", "source"), ("edited_image", "target")]:
        with Image.open(io.BytesIO(row[field]["bytes"])) as image:
            image.convert("RGB").save(root / f"{i:03d}-{label}.png")
    metadata.append(
        {
            "prompt": row["edit_prompt"],
            "edit_image": [f"{i:03d}-source.png"],
            "image": f"{i:03d}-target.png",
        }
    )
for n in [4, 8, 32, 128]:
    (root / f"train-{n}.json").write_text(json.dumps(metadata[:n], indent=2))
files = [
    {
        "path": p.name,
        "bytes": p.stat().st_size,
        "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
    }
    for p in sorted(root.iterdir())
    if p.is_file()
]
(root / "manifest.json").write_text(
    json.dumps(
        {
            "repository": repo,
            "revision": info.sha,
            "shard": shard,
            "rows": list(range(len(rows))),
            "files": files,
        },
        indent=2,
    )
)
print("Prepared", len(rows), "pairs", flush=True)
