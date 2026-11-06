# Frozen-condition reuse in image SFT

Reproducibility evidence for the Relax Task17 contribution by WYXNICK.
The implementation shares one bounded GPU cache between SenseNova and
Qwen-Image-Edit. Repeated conditions are borrowed directly from device memory,
preserving the complete training update while removing repeated frozen encoding.

| Two H100 SXM 80GB GPUs | SenseNova | Qwen-Image-Edit |
| --- | ---: | ---: |
| Median steady end-to-end throughput gain | **14.59%** | **9.59%** |
| Submission-to-completion throughput gain | **5.39%** | **6.32%** |
| Loss / reduced gradients / FP32 updates / RNG | Exact | Exact |
| Reusable encoder calls in the profiled hot window | **0** | **0** |

The repeated benchmark consists of 24 production Ray/FSDP jobs: two models,
high and low reuse, and three paired repeats per mode. Four supplementary
prefetch runs also reproduce the original full training checkpoints exactly.

- [Acceptance report](REPORT.md): workload selection, every timing scope,
  dispersion, low-reuse overhead, resources, cold-fill cost and numerical checks.
- [Reproduction](REPRODUCE.md): pinned model/data versions and runnable tools.
- [Measurement summaries](summaries/): performance, resources and CUDA profiles.
- [Raw evidence archive](artifacts/validation-evidence.tar.gz): every reported
  production repeat, detailed numerical comparisons and lifecycle evidence.
- [Archive checksum](artifacts/validation-evidence.tar.gz.sha256) and
  [file manifest](artifacts/evidence-manifest.json).

This evidence is maintained on a separate branch of the contributor's fork.
It is excluded from the Relax implementation PR. Infrastructure paths and
hostnames are normalized for publication; numerical measurements are preserved.
