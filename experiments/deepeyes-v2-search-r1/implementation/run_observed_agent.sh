#!/usr/bin/env bash
set -eu
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"
export PYTHONNOUSERSITE=1
mkdir -p "${AGENT_DEBUG_LOG_DIR}"
exec >>"${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID}.log" 2>&1
exec "${DEEPEYES_V2_APP_PYTHON}" "${EXPERIMENT_OPS_DIR}/observed_agent.py" \
    --input-json "${RELAX_INPUT_JSON}" --output-json "${RELAX_OUTPUT_JSON}"
