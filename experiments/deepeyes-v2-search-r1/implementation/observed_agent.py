"""Experiment-only observer; delegates behavior and rewards to the V2 app."""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

from app import agent, env_deepeyes_v2

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

original_run = agent.run_session
original_search = env_deepeyes_v2.search
original_dispatch = env_deepeyes_v2.DeepEyesV2Env._dispatch_search
counts = defaultdict(float)
events = []


def observed_search(*args, **kwargs):
    result = original_search(*args, **kwargs)
    if isinstance(result, dict):
        counts["text_search_empty"] += int(not result.get("data"))
    events.append({"query": args[0] if args else kwargs.get("query"), "result": result})
    return result


def observed_dispatch(self, name, arguments):
    prefix = "text_search" if name == "search" else "image_search"
    counts[prefix + "_calls"] += 1
    start = time.monotonic()
    try:
        result = original_dispatch(self, name, arguments)
        counts[prefix + "_success"] += int(result.get("status") == "success")
        counts[prefix + "_failure"] += int(result.get("status") != "success")
        return result
    except Exception:
        counts[prefix + "_failure"] += 1
        raise
    finally:
        counts[prefix + "_seconds"] += time.monotonic() - start


def compact_messages(messages):
    result = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [part if part.get("type") == "text" else {"type": "image_omitted"} for part in content]
        result.append({"role": message.get("role"), "content": content})
    return result


async def observed_run(messages, metadata):
    counts.clear()
    events.clear()
    start = time.monotonic()
    for prefix in ("text_search", "image_search"):
        for suffix in ("calls", "success", "failure", "seconds"):
            counts[prefix + "_" + suffix] = 0.0
    counts["text_search_empty"] = 0.0
    output = await original_run(messages, metadata)
    output["metadata"].update(counts)
    output["metadata"]["session_wall_seconds"] = time.monotonic() - start
    index = str(metadata.get("data_index") or metadata.get("index") or "?")
    output["metadata"]["source_id"] = index
    session_id = os.environ["RELAX_SESSION_ID"]
    output["metadata"]["observer_session_id"] = session_id
    try:
        run_dir = Path(os.environ["EXPERIMENT_RUN_DIR"])
        fixed = json.loads(Path(os.environ["EXPERIMENT_DATA_MANIFEST"]).read_text())["fixed_case_ids"]
        if index in fixed:
            directory = run_dir / "trajectories" / index
            directory.mkdir(parents=True, exist_ok=True)
            record = {
                "source_id": index,
                "session_id": session_id,
                "metadata": output["metadata"],
                "messages": compact_messages(messages),
                "text_search_events": events,
            }
            (directory / (session_id + ".json")).write_text(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as exc:
        logger.warning("Experiment trajectory recording failed; agent result preserved: %s", exc)
    return output


if __name__ == "__main__":
    env_deepeyes_v2.search = observed_search
    env_deepeyes_v2.DeepEyesV2Env._dispatch_search = observed_dispatch
    agent.run_session = observed_run
    agent.main()
