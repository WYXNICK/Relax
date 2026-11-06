# Task 17 acceptance report

**Status:** all 24 production performance jobs and 19 numerical, resume/export, CUDA-trace, eviction, trajectory and lifecycle reports passed on two H100 SXM 80GB GPUs.

Requirement: [official Task 17](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-17). The contribution is independent of competing PR implementations.

## Implementation and scope

The shared cache is a process-local GPU LRU bounded by storage bytes and entry count. It borrows immutable tensors without host staging, serialization or background services. Each loaded immutable model owns its cache; resume starts empty. Adapters declare conditioning inputs and frozen dependencies. Existing rank agreement prevents a local hit from skipping nested FSDP collectives on only one rank.

SenseNova reuses frozen vision/understanding K/V and positional IDs. The new minimal Qwen-Image-Edit SFT adapter reuses prompt embeddings and deterministic reference VAE latents. Target posterior samples, noise, timesteps, generation/DiT, backward and optimizer updates remain live. Cache defaults to off.

The tested upstream base is `9213c4b716a8e2351ff1b6a31cbf9e03256b9f66`. The exact performance source is preserved in `source/implementation-2026-11-06.patch` and `source/code-manifest-2026-11-06.json`. The subsequent prefetch extension enables the existing CPU prefetcher for Qwen-Image-Edit. Its four additional production runs match the original checkpoints exactly; the repeated performance estimates below retain the original no-prefetch configuration.

## Fixed workload

- One node, two H100 SXM 80GB GPUs with NVLink; world size 2, global batch 4, microbatch 1, two microbatches per rank.
- Rank-8/alpha-8 DiT/generation LoRA, FP32 masters, BF16 computation, AdamW lr 1e-4, weight decay 0.01, clipping 1.0, deterministic native Torch attention, gradient checkpointing, no image prefetch.
- `max_pixels=262144`, original aspect ratio with adapter patch alignment. All runs use the same 32 steps and a committed checkpoint at step 32.
- High reuse: first eight real image-edit rows repeated for 16 epochs. One pass: first 128 rows, one epoch. Selection is the first rows of a pinned first parquet shard, not selected for measured speedup.
- Cache on: 2 GiB and 128 entries per rank. Cache off: zero bytes. Each model/regime has three repeats per mode, ordered off/on, on/off, off/on.
- High-reuse warmup is 12 steps, one-pass startup window is four. A CPU sampler audit before inspecting timing or starting cache-on found both ranks finish seeing all eight rows at step 11; the documented warmup amendment preserves all 32 steps of the already-running first baseline.
- The user-provided idle keeper is stopped and awaited before the first training step. Every performance job certifies this with recorded timestamps. Profiling and synthetic numerical diagnostics are excluded from throughput.

Pinned model/data revisions and full commands are in [REPRODUCE.md](REPRODUCE.md), the asset manifests and each run's `job.json` / `submission.log`. The production package lock is `runner/ray-python312-freeze.txt`.

## Production throughput

Values below are medians of three successful runs per mode. Samples/s counts the global effective batch once. The steady interval includes CPU preparation, TQ/Controller gaps, forward, backward, optimizer, metrics and the final checkpoint. Whole-job time runs from submission through terminal completion.

| Model / workload | Off samples/s | On samples/s | Steady change | Off job s | On job s | Whole-job throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SenseNova / high reuse | 1.3142 | 1.5060 | +14.59% | 238.116 | 225.935 | +5.39% |
| SenseNova / one pass | 1.3341 | 1.2891 | -3.38% | 242.128 | 235.667 | +2.74% |
| Qwen-Image-Edit / high reuse | 0.9092 | 0.9965 | +9.59% | 275.371 | 259.000 | +6.32% |
| Qwen-Image-Edit / one pass | 0.9215 | 0.9110 | -1.14% | 267.456 | 269.168 | -0.64% |

SenseNova exceeds the task's 10% steady-E2E bonus threshold. Qwen has an independent positive comparison. Neither model benefits from caching a one-pass dataset with zero reuse; the feature remains opt-in. SenseNova's positive one-pass whole-job number is dominated by startup variation and is not evidence of a caching benefit.

### Dispersion

| Model / workload | Off samples/s range; sample SD | On samples/s range; sample SD |
| --- | --- | --- |
| SenseNova / high | 1.2709–1.3169; 0.0258 | 1.4923–1.5102; 0.0093 |
| SenseNova / low | 1.2908–1.3342; 0.0250 | 1.2719–1.2938; 0.0116 |
| Qwen / high | 0.9089–0.9154; 0.0037 | 0.9699–1.0337; 0.0320 |
| Qwen / low | 0.9070–0.9256; 0.0097 | 0.8986–0.9225; 0.0120 |

All individual repeats, whole-job variation and per-rank counters remain in the four `results/*-summary.json` files. No slow repeat or checkpoint step is removed. Percentages from competing PRs on H20/H200 or different workloads are not directly comparable to these H100 measurements.

### Cold fill and break-even

There is no separate offline preprocessing job. Conditions are admitted on demand during real training. In both high-reuse runs, effective reuse first occurs at step 3 and both ranks hold all eight conditions after step 11. The fixed cold window ends at step 12: 48 global samples, or six epochs. It includes useful training and early hits, not just dictionary insertion.

| Model | Off cold-window s | On cold-window s | Net change s | Off full-training s | On full-training s |
| --- | ---: | ---: | ---: | ---: | ---: |
| SenseNova | 38.413 | 35.968 | -2.445 | 99.267 | 88.926 |
| Qwen | 56.735 | 52.603 | -4.132 | 144.309 | 132.865 |

Both observed median trajectories have already repaid fill cost by the end of that 48-sample window; no additional steady steps are needed in the cold-overhead estimate. The raw first sustained crossing may precede any cache hit because of timing noise and must not be interpreted as causal savings before reuse. Full cumulative trajectories and startup latency are retained. These are workload-specific observations, not universal break-even guarantees.

## Numerical and training-state equivalence

- Both real pretrained models passed exact loss, full accumulated/FSDP-reduced per-parameter gradient, actual FP32 optimizer-delta and RNG comparisons for baseline repetition, cold admission, hot hits, asymmetric rank misses and clear/recompute. All compared values were finite; maximum absolute errors were zero. Hot borrowed tensors remained unchanged and recomputation reproduced the same conditions.
- Qwen's final text-output LoRA A/B parameters are structurally unused because its last block discards the text stream. The diagnostic requires exactly these two absent gradients, compares their zero updates, and requires nonzero training gradients/updates elsewhere. It checks all 960 parameter updates; SenseNova checks all 588.
- Cache-on steps 1–4 followed by cache-off resume through step 8 matched uninterrupted cache-off training exactly for both models: 2,358 SenseNova and 3,840 Qwen checkpoint tensors plus optimizer/trainer metadata, contract and every rank's Python/NumPy/Torch/CUDA RNG. Qwen also passed the reverse cache-off-to-on resume.
- Export/load passed for all 588 SenseNova and 960 Qwen LoRA tensors. Qwen used the public Diffusers load API with explicit offline `weight_name` and preserved scaling. CPU regressions cover non-unit alpha.
- CPU regression suites and full-repository pre-commit checks passed; later affected-scope checks also passed. Raw GPU reports and pre-commit logs are under `results/`. The historical Nov 5 CPU record is explicitly marked as superseded.

## Resources

The cache owns no shared-memory or disk artifacts. High-reuse residency is 709,165,056 bytes per SenseNova rank and 24,000,512 bytes per Qwen rank. Both finish with 64 lookups, 56 local hits, 53 effective hits and 11 encodes per rank; the steady window is entirely effective hits. One-pass runs have zero hits. SenseNova evicts 40 entries per rank and never exceeds the 2 GiB byte bound.

Across all paired runs, the maximum allocated/reserved GPU memory per device was 33.113/36.488 GiB for SenseNova and 58.809/60.338 GiB for Qwen. Shared-memory peaks were 64.293/64.301 MiB. Container charged-memory peaks were about 230.4 GiB and include reclaimable file cache; this is not a minimum RAM requirement or cache allocation. Process RSS sums are reported separately because they can double-count shared pages.

CPU synchronous preparation time increased by about 0.002 seconds per step in the median comparisons. This metric includes image decode/resize/crop and key lookup, and excludes GPU encoding and TQ/rank agreement. Raw TensorBoard events and `preparation-summary.json` are preserved.

Filesystem-wide used space at job boundaries ranged from 1,311.7 to 1,462.9 GiB; this measures the entire backing filesystem, not this job or cache. The task directory contains deliberately retained model snapshots, backup archives, checkpoints and traces. A separate final directory inventory records their sizes. Cache-owned disk bytes are zero. Per-run resource samples and startup measurements are in `results/resources-summary.json` and each `resources.jsonl`.

## Independent diagnostics

Four independent rank-zero traces passed event-level validation. Each captures 15 steps (30 local samples), with a hot window of steps 13–15 (six local samples). Both SenseNova condition units and Qwen prompt/reference encoders go from six hot calls online to zero with reuse. Qwen target VAE remains at six hot calls in both modes; across the full trace it executes all 30 times. Cold misses remain visible (11 calls per reusable encoder with caching versus 30 online).

| Rank-zero hot-window H2D bytes | Cache off | Cache on |
| --- | ---: | ---: |
| SenseNova | 47,225,788 | 28,917,444 |
| Qwen-Image-Edit | 95,630,964 | 49,141,836 |

Kineto CUDA kernels and H2D events are correlated to named CPU ranges by External id. Encoder, backward, optimizer and other forward/framework stage counts, kernel-duration sums and bytes are in `summaries/profile-*.json`. Kernel sums can overlap across streams; they are not end-to-end latency. Other forward/framework includes DiT and distributed/runtime work, not isolated DiT compute. Traces are independent of the 24 timing jobs.

A separate eight-step SenseNova run with a 128 MiB cap retained one entry and evicted 13 times on each rank. Peak backing storage was 89,118,720 bytes, below the cap throughout. No serialized cache artifacts are created.

The eviction run's final checkpoint matched uninterrupted online training exactly across all 2,358 tensors and metadata/RNG. Complete 32-step off/on checkpoints also matched exactly for all four model/workload combinations: 2,358 tensors per SenseNova pair and 3,840 per Qwen pair, plus optimizer/trainer state and every rank's RNG. These compare the first predeclared repeat, not a selected favorable trajectory.

A populated-cache malformed-image run failed at the intended third-step sample with `UnidentifiedImageError`. Both FSDP processes exited, their actors became DEAD and their placement group became REMOVED. Two GPUs returned to Ray, shared memory returned to its 1,347,584-byte baseline, and no Serve application or TQ controller survived. Only the completed step-two checkpoint was committed.

Cancellation after at least four completed steps also passed. The job reported STOPPED, original worker PIDs disappeared, actors became DEAD and the placement group became REMOVED; both GPUs returned to Ray, with no surviving Serve application or TQ controller. Shared memory returned to 1,347,584 bytes. No incomplete checkpoint was marked committed. These deliberately FAILED/STOPPED jobs are successful lifecycle diagnostics, excluded from throughput measurements. Raw before/after snapshots are retained in `results/cleanup-*/cleanup.json`.

## Evidence and reproduction

The evidence archive contains raw production training logs, submission arguments,
resource samples, TensorBoard scalar exports, complete numerical comparisons,
checkpoint/resume/export comparisons, lifecycle records and profiler summaries.
All 24 performance repeats and four supplementary prefetch runs are included.
See [REPRODUCE.md](REPRODUCE.md) for extraction and rerun commands. Machine-specific
paths, hostnames and private addresses are normalized; numerical measurements
and experimental settings are preserved. The recorded CPU/GPU scope is
single-node two-GPU LoRA SFT.
