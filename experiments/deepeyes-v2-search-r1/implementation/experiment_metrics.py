"""Small additions through Relax's supported logging hooks; retain native
logs."""

import json
import os
from collections import defaultdict
from pathlib import Path

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def record(rollout_id, args, samples, phase):
    from relax.utils import tracking_utils
    from relax.utils.metrics.metric_utils import compute_rollout_step

    flat = [sample for group in samples for sample in (group if isinstance(group, list) else [group])]
    metrics, groups = {}, defaultdict(list)
    for prefix in ("text_search", "image_search"):
        calls = sum(s.metadata.get(prefix + "_calls", 0) for s in flat)
        failures = sum(s.metadata.get(prefix + "_failure", 0) for s in flat)
        seconds = sum(s.metadata.get(prefix + "_seconds", 0) for s in flat)
        metrics[prefix + "/calls"] = calls
        metrics[prefix + "/calls_per_trajectory"] = calls / max(1, len(flat))
        metrics[prefix + "/used_fraction"] = sum(s.metadata.get(prefix + "_calls", 0) > 0 for s in flat) / max(
            1, len(flat)
        )
        if calls:
            metrics[prefix + "/failure_rate"] = failures / calls
            metrics[prefix + "/seconds_per_call"] = seconds / calls
            if prefix == "text_search":
                metrics[prefix + "/empty_rate"] = sum(s.metadata.get("text_search_empty", 0) for s in flat) / calls
    for sample in flat:
        reward = sample.reward
        value = reward.get("score") if isinstance(reward, dict) else reward
        if value is not None:
            groups[sample.group_index].append(value)
    usable = [values for values in groups.values() if len(values) > 1]
    if usable:
        metrics["reward/constant_group_fraction"] = sum(max(x) == min(x) for x in usable) / len(usable)
    step = compute_rollout_step(args, rollout_id)
    tracking_utils.log_direct(args, step, {f"{phase}/tools/{key}": value for key, value in metrics.items()})
    directory = Path(os.environ["EXPERIMENT_RUN_DIR"]) / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "tool-samples.jsonl").open("a") as f:
        for sample in flat:
            metadata = sample.metadata
            selected = {
                key: value
                for key, value in metadata.items()
                if key.startswith(("text_search_", "image_search_"))
                or key
                in (
                    "source_id",
                    "observer_session_id",
                    "data_source",
                    "stop_reason",
                    "last_error",
                    "session_wall_seconds",
                )
            }
            f.write(
                json.dumps(
                    {
                        "rollout_id": rollout_id,
                        "step": step,
                        "phase": phase,
                        "sample_index": sample.index,
                        "group_index": sample.group_index,
                        "reward": sample.reward,
                        "metadata": selected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def train(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    try:
        record(rollout_id, args, samples, "rollout")
    except Exception as exc:
        logger.warning("Experiment metrics recording failed; native logging continues: %s", exc)
    return False


def evaluate(rollout_id, args, data, extra_metrics):
    for name, value in data.items():
        try:
            record(rollout_id, args, value.get("samples") or [], "eval/" + name)
        except Exception as exc:
            logger.warning("Experiment eval recording failed; native logging continues: %s", exc)
    return False
