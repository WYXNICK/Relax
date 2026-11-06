# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real-model FSDP numerical acceptance; run with torchrun, never as a speed
test.

Reports per-parameter reduced-gradient and actual FP32 update errors. CPU
snapshots and synchronization are deliberately outside production training. See
REPRODUCE.md for the separate Ray E2E protocol.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from relax.backends.fsdp.lora import inject_lora_adapter
from relax.backends.fsdp.runtime import fsdp2_wrap
from relax.backends.fsdp.sft import clip_sharded_gradients, seed_training
from relax.engine.sft.image import ImageBatchSampler, ImageEditDataset
from relax.models.frozen_conditions import FrozenConditionCache, agree_condition_request


def compare_tensors(
    reference: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], group: Any
) -> dict:
    """Aggregate shard norms, never average per-rank relative errors."""
    if reference.keys() != actual.keys():
        raise ValueError("Parameter/gradient keysets differ")
    names = sorted(reference)
    sums, maxima = [], []
    for name in names:
        a, b = reference[name].double(), actual[name].double()
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {name}")
        difference = b - a
        sums.append([a.square().sum(), difference.square().sum()])
        maxima.append(
            [
                difference.abs().max() if difference.numel() else torch.tensor(0.0),
                float(not torch.equal(a, b)),
                float(not (torch.isfinite(a).all() and torch.isfinite(b).all())),
            ]
        )
    sums, maxima = (
        torch.tensor(sums, dtype=torch.float64),
        torch.tensor(maxima, dtype=torch.float64),
    )
    dist.all_reduce(sums, group=group)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
    result = {}
    for index, name in enumerate(names):
        norm, error = sums[index].sqrt().tolist()
        max_abs, unequal, nonfinite = maxima[index].tolist()
        result[name] = {
            "reference_l2": None if nonfinite else norm,
            "error_l2": None if nonfinite else error,
            "max_abs": None if nonfinite else max_abs,
            "relative_l2": None
            if nonfinite
            else error / norm
            if norm
            else (0.0 if error == 0 else None),
            "exact": not unequal,
            "finite": not nonfinite,
        }
    return result


def passes(values: dict[str, dict], *, exact: bool, atol: float = 0.0) -> bool:
    return all(
        value["finite"]
        and (
            value["exact"]
            if exact
            else value["relative_l2"] is not None
            and value["relative_l2"] <= 0.01
            and value["max_abs"] <= atol
        )
        for value in values.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON with model_path, model_adapter_path, prompt_data, i2i_sft_config",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument(
        "--atol",
        type=float,
        help="Predeclared near-zero max absolute error ceiling for non-exact runs",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require exact equality; never relax after inspecting results",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable Torch deterministic algorithms before model load",
    )
    options = parser.parse_args()
    if options.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if options.microbatches <= 0:
        parser.error("microbatches must be positive")
    if not options.exact and (
        options.atol is None or not math.isfinite(options.atol) or options.atol < 0
    ):
        parser.error(
            "Non-exact acceptance requires an explicit finite nonnegative --atol"
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    args = argparse.Namespace(**json.loads(options.config.read_text()))
    args.fsdp_trainable_mode, args.fsdp_param_dtype = "lora", "bf16"
    seed_training(42)
    module, name = args.model_adapter_path.rsplit(".", 1)
    adapter = getattr(importlib.import_module(module), name)()
    model = adapter.load_train_model(args)
    inject_lora_adapter(
        model, rank=8, alpha=8, target_modules=adapter.lora_target_modules
    )
    adapter.prepare_for_sharding(model)
    adapter.validate_condition_cache(model)
    model = fsdp2_wrap(model, master_dtype="fp32")
    adapter.on_device_ready(model, device)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters or any(
        "lora_" not in name or p.dtype != torch.float32
        for name, p in parameters.items()
    ):
        raise ValueError("Acceptance requires nonempty FP32 LoRA masters")
    initial = {
        name: p.to_local().detach().cpu().clone() for name, p in parameters.items()
    }
    dataset = ImageEditDataset(args, adapter)
    ids = ImageBatchSampler(len(dataset.rows), options.microbatches * world, 42).batch(
        0
    )[rank::world]
    cache = FrozenConditionCache(options.cache_bytes, options.microbatches)

    expected_unused = set()
    if adapter.family == "qwen_image_edit":
        last = len(model.transformer.transformer_blocks) - 1
        expected_unused = {
            f"transformer.transformer_blocks.{last}.attn.to_add_out.lora_{side}.default.weight"
            for side in ("A", "B")
        }

    def update(enabled: bool) -> dict:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.to_local().copy_(initial[name])
        model.condition_cache = cache if enabled else None
        optimizer = torch.optim.AdamW(
            parameters.values(), lr=1e-4, weight_decay=0.01, foreach=False
        )
        optimizer.zero_grad(set_to_none=True)
        seed_training(42 + rank)
        loss_sum = torch.zeros((), device=device)
        for sample_id in ids:
            data = dataset[sample_id]
            if enabled:
                request = cache.lookup(model.condition_key(data))
                hits = [None] * world
                dist.all_gather_object(hits, request.tensors is not None, group=control)
                data = {
                    **data,
                    "condition_request": agree_condition_request(request, all(hits)),
                }
                del request
            loss = model(data) / len(ids)
            loss.backward()
            loss_sum += loss.detach()
        dist.all_reduce(loss_sum, group=group)
        # These are accumulated AND FSDP-reduced gradients, before clipping.
        gradients = {
            name: p.grad.to_local().detach().cpu().clone()
            if p.grad is not None
            else torch.zeros_like(initial[name])
            for name, p in parameters.items()
        }
        all_present = torch.tensor(
            int(all(p.grad is not None for p in parameters.values()))
        )
        dist.all_reduce(all_present, op=dist.ReduceOp.MIN, group=control)
        unused = {name for name, p in parameters.items() if p.grad is None}
        expected_presence = torch.tensor(int(unused == expected_unused))
        dist.all_reduce(expected_presence, op=dist.ReduceOp.MIN, group=control)
        clip_sharded_gradients(list(parameters.values()), 1.0, group)
        optimizer.step()
        updates = {
            name: p.to_local().detach().cpu() - initial[name]
            for name, p in parameters.items()
        }
        return {
            "loss": {"loss": (loss_sum / world).cpu()},
            "all_gradients_present": bool(all_present),
            "expected_gradient_presence": bool(expected_presence),
            "gradient": gradients,
            "gradient_present": {
                name: torch.tensor(p.grad is not None) for name, p in parameters.items()
            },
            "update": updates,
            "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        }

    report = {
        "world_size": world,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "config": vars(args),
        "exact_required": options.exact,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "absolute_tolerance": options.atol,
        "structurally_unused_parameters": sorted(expected_unused),
        "comparisons": {},
        "passed": True,
    }
    try:
        baseline = update(False)
        cold_conditions = None
        for label in ("baseline_repeat", "cold", "hot", "mixed_miss", "cold_again"):
            if label == "mixed_miss" and rank == 0 or label == "cold_again":
                cache.clear()
            before_hits = cache.hits
            actual = update(label != "baseline_repeat")
            comparison = {
                kind: compare_tensors(baseline[kind], actual[kind], control)
                for kind in ("loss", "gradient", "gradient_present", "update", "rng")
            }
            comparison["passed"] = all(
                passes(
                    comparison[kind],
                    exact=options.exact or kind in ("rng", "gradient_present"),
                    atol=options.atol,
                )
                for kind in ("loss", "gradient", "gradient_present", "update", "rng")
            )
            update_norms = [
                value["reference_l2"] for value in comparison["update"].values()
            ]
            comparison["baseline_update_nonzero"] = any(
                norm is not None and norm > 0 for norm in update_norms
            )
            comparison["baseline_gradient_nonzero"] = any(
                value["reference_l2"] is not None and value["reference_l2"] > 0
                for value in comparison["gradient"].values()
            )
            comparison["all_gradients_present"] = (
                baseline["all_gradients_present"] and actual["all_gradients_present"]
            )
            comparison["passed"] &= (
                comparison["baseline_update_nonzero"]
                and comparison["baseline_gradient_nonzero"]
                and baseline["expected_gradient_presence"]
                and actual["expected_gradient_presence"]
            )
            if label == "hot":
                all_hit = torch.tensor(int(cache.hits - before_hits == len(ids)))
                dist.all_reduce(all_hit, op=dist.ReduceOp.MIN, group=control)
                comparison["all_hot_requests_hit"] = bool(all_hit)
                comparison["passed"] &= bool(all_hit)
            if label in ("cold", "hot", "cold_again"):
                conditions = {
                    (key, name): tensor.detach().cpu().clone()
                    for key, (tensors, _) in cache._entries.items()
                    for name, tensor in tensors.items()
                }
                if label == "cold":
                    cold_conditions = conditions
                else:
                    equal = conditions.keys() == cold_conditions.keys() and all(
                        torch.equal(tensor, cold_conditions[key])
                        for key, tensor in conditions.items()
                    )
                    agreement = torch.tensor(int(equal))
                    dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=control)
                    comparison["conditions_exact"] = bool(agreement)
                    comparison["passed"] &= bool(agreement)
            comparison["cache_rank0"] = cache.metrics()
            report["comparisons"][label] = comparison
            report["passed"] &= comparison["passed"]
        if rank == 0:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    finally:
        cache.clear()
        dist.destroy_process_group()
    if not report["passed"]:
        raise SystemExit(
            "Numerical acceptance failed; inspect the per-parameter report"
        )


if __name__ == "__main__":
    main()
