#!/usr/bin/env bash

# ==============================================================================
# SOVEREIGN PROBABILITY ENGINE — STAGE 8: REAPER CONTAINER KILL-SWITCH
# File: 06_survival_watchdog/reaper_controller/container_kill_switch.sh
# Purpose: Enforces the strict "Earn or Die" survival mandate. If Net Liquid
# Balance <= $0.00, archives the last 500 log lines, writes an
# immutable Death Certificate, and executes `docker stop & rm` on
# `sovereign_survival_agent` (unless DRY_RUN=1).
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

TARGET_CONTAINER="${1:-${TARGET_AGENT_CONTAINER:-sovereign_survival_agent}}"
REASON="${2:-NET_LIQUID_BALANCE_EXHAUSTED}"
DRY_RUN="${DRY_RUN:-1}"
TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORENSIC_DIR="${REPO_ROOT}/state/forensics/post_mortems"

mkdir -p "${FORENSIC_DIR}"

MORTEM_FILE="${FORENSIC_DIR}/death_certificate_${TIMESTAMP_UTC//:/-}.json"

if [[ "${DRY_RUN}" == "0" ]] && command -v docker >/dev/null 2>&1 && [[ -S /var/run/docker.sock ]]; then
    if docker ps -a --format '{{.Names}}' | grep -q "^${TARGET_CONTAINER}$"; then
        docker logs --tail 500 "${TARGET_CONTAINER}" > "${FORENSIC_DIR}/${TARGET_CONTAINER}_final.log" 2>&1 || true
        docker stop -t 15 "${TARGET_CONTAINER}" || true
        docker rm -f "${TARGET_CONTAINER}" || true
        ACTION_TAKEN="DOCKER_STOP_AND_RM_EXECUTED"
    else
        ACTION_TAKEN="CONTAINER_ALREADY_ABSENT"
    fi
else
    ACTION_TAKEN="DRY_RUN_DEATH_CERTIFICATE_RECORDED"
fi

cat > "${MORTEM_FILE}" <<JSON
{
    "event": "AGENT_TERMINATED_BY_REAPER",
    "target_container": "${TARGET_CONTAINER}",
    "reason": "${REASON}",
    "action_taken": "${ACTION_TAKEN}",
    "timestamp_utc": "${TIMESTAMP_UTC}",
    "survival_rule": "STRICT_EARN_OR_DIE_NAV_FLOOR_BREACHED"
}
JSON

cat "${MORTEM_FILE}"