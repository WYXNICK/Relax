# Reproduce the measurements

## Inspect the recorded evidence

Download this directory from the pinned evidence commit. From this directory:

```bash
cd artifacts
sha256sum -c validation-evidence.tar.gz.sha256
cd ..
tar -xzf artifacts/validation-evidence.tar.gz
python acceptance/summarize_frozen_conditions.py results/sensenova-high-manifest.json --output /tmp/sensenova-high.json
python acceptance/summarize_frozen_conditions.py results/qwen-high-manifest.json --output /tmp/qwen-high.json
python acceptance/summarize_frozen_conditions.py results/sensenova-low-manifest.json --output /tmp/sensenova-low.json
python acceptance/summarize_frozen_conditions.py results/qwen-low-manifest.json --output /tmp/qwen-low.json
```

Each group contains three paired repeats. The summarizer retains checkpoint
steps and Controller gaps, counts the global batch once, sums intervals per
rank and uses the largest rank total. It reports medians, sample standard
deviations, ranges, cold-fill costs and cumulative training savings.

`results/*-numerics.json` contains per-parameter gradient and FP32 update checks,
including finite/zero-reference checks and each rank's RNG comparison.
`results/*-comparison.json` records full-checkpoint comparisons.
`results/cleanup-*/cleanup.json` records failure/cancellation cleanup.
`prefetch-results/` contains all four additional 32-step runs and their exact
checkpoint comparisons. Profiler summaries are in `summaries/profile-*.json`;
`acceptance/analyze_profile.py` analyzes a locally captured compressed Kineto
trace separately from timing runs.

## Models, data and source

| Asset | Revision |
| --- | --- |
| `sensenova/SenseNova-U1.5-8B-MoT` | `9feeeab8a2792514d109cd34589342a2cc1d4ab2` |
| `Qwen/Qwen-Image-Edit` | `ac7f9318f633fc4b5778c59367c8128225f1e3de` |
| `timbrooks/instructpix2pix-clip-filtered` | `aa665b890915f7a42f8615bee868a9f3447e178f` |

`assets/` records the exact model/data file hashes. Data selection is the first
128 rows of the first sorted parquet shard. The high-reuse workload repeats
the first eight rows; the one-pass workload uses all 128. `runner/prepare_data.py`
creates RGB PNG pairs without resizing; the adapters perform training preprocessing.

The repeated performance source is preserved in
`source/implementation-2026-11-06.patch`, based on upstream commit
`9213c4b716a8e2351ff1b6a31cbf9e03256b9f66`. Verify its files against
`source/code-manifest-2026-11-06.json`. `source/submitted-source.json` identifies
the submitted source and its relationship to this measured snapshot.

## Run the experiment

Use an exclusive two-GPU node with the Relax FSDP runtime. The tested production
environment is Python 3.12.16, Torch 2.9.1/cu126, Diffusers 0.37.0,
Transformers 5.12.1 and PEFT 0.20.0; `runner/ray-python312-freeze.txt` records the
complete environment. The numerical diagnostic used Python 3.11 and the same
model-library versions. Point `RELAX_REPO` to the Relax checkout and make its
Megatron dependency available to the shared argument parser.

The supplied runner uses `/tmp/relax-task17` for the environment, downloaded
assets and run directories. Adapt those paths to your installation. Its
loopback Ray/NCCL configuration is for this single-node experiment.

```bash
python runner/download.py
python runner/prepare_data.py
export RELAX_REPO=/workspace/task17/Relax
export PYTHONPATH="$RELAX_REPO:/tmp/relax-task17/dependencies/Megatron-LM"
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export RAY_ADDRESS=auto
export NO_PROXY=127.0.0.1,localhost
ray start --head --node-ip-address=127.0.0.1 --dashboard-host=127.0.0.1 --num-gpus=2 --object-store-memory=4294967296
python runner/run-performance.py
```

Use a fresh scratch output for new measurements. The public runner removes the
original worker's idle-only GPU-keeper hook and requires exclusive GPUs with
no other workload during measurement. Every recorded performance job includes
timestamps certifying that the original keeper had exited before training.
The training configuration and serial paired schedule remain unchanged.

For real-model numerical checks, extract the `config` object from the matching
`results/*-numerics.json`, adapt its model/data paths and run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  acceptance/validate_frozen_conditions.py --config model-config.json \
  --output numerics.json --microbatches 2 --cache-bytes 2147483648 --exact --deterministic
```

Use `acceptance/compare_checkpoints.py` for complete training-state comparisons
and the supplied export-load scripts for adapter loading. Capture CUDA traces
in separate runs with the existing Relax profiler controls so profiling does
not enter throughput measurements.
