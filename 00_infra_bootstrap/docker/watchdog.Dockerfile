# syntax=docker/dockerfile:1.7
# ==============================================================================
# SOVEREIGN PROBABILITY ENGINE — STAGE 1: INFRASTRUCTURE & CREDIT BOOTSTRAP
# File: 00_infra_bootstrap/docker/watchdog.Dockerfile
# Purpose: Isolated supervisor runtime for the Sovereign Survival Agent.
#
# This image provides the Python/Docker client environment and kill-switch
# tooling. The application launched by CMD must implement NAV polling,
# on-chain valuation, compute-burn accounting, and any clone/mitosis policy.
# ==============================================================================

ARG PYTHON_BASE_IMAGE="python:3.11-slim-bookworm"

# ------------------------------------------------------------------------------
# STAGE 1: PYTHON DEPENDENCY BUILDER
# ------------------------------------------------------------------------------
FROM ${PYTHON_BASE_IMAGE} AS watchdog_builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/watchdog_venv
ENV PATH="/opt/watchdog_venv/bin:${PATH}"

# Docker SDK, Polygon/Web3 and CLOB dependencies, plus logging/telemetry.
RUN pip install --upgrade \
      "pip>=24.2" \
      "setuptools>=72.1.0" \
      "wheel>=0.44.0" \
    && pip install \
      "docker==7.1.0" \
      "web3==7.6.0" \
      "eth-account==0.13.4" \
      "eth-abi==5.1.0" \
      "py-clob-client==0.18.0" \
      "httpx==0.28.1" \
      "aiohttp==3.11.11" \
      "tenacity==9.0.0" \
      "pydantic==2.10.4" \
      "orjson==3.10.12" \
      "psutil==6.1.1" \
      "prometheus-client==0.21.1" \
      "rich==13.9.4" \
      "loguru==0.7.3"

# Remove bytecode/cache directories from the copied virtual environment.
RUN find /opt/watchdog_venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/watchdog_venv -type f -name '*.pyc' -delete

# ------------------------------------------------------------------------------
# STAGE 2: REAPER SUPERVISOR RUNTIME
# ------------------------------------------------------------------------------
FROM ${PYTHON_BASE_IMAGE} AS reaper_watchdog_runtime

LABEL org.opencontainers.image.title="Sovereign Probability Engine - Reaper Watchdog" \
      org.opencontainers.image.description="Supervisor tooling for NAV monitoring and survival policy" \
      org.opencontainers.image.version="1.0.0-watchdog"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/watchdog_venv/bin:${PATH}" \
    WATCHDOG_POLL_INTERVAL_SEC="60" \
    TARGET_AGENT_CONTAINER="sovereign_survival_agent" \
    SURVIVAL_DEATH_FLOOR_USD="0.00" \
    MITOSIS_CLONE_TRIGGER_USD="50.00" \
    MITOSIS_SEED_ALLOCATION_USD="5.00" \
    COMPUTE_HOURLY_BURN_USD="0.0045" \
    POLYGON_CHAIN_ID="137"

# docker.io supplies the Docker CLI; docker.sock must be mounted at runtime
# for host-container control. Mounting it grants powerful host-level access.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      coreutils \
      curl \
      docker.io \
      grep \
      jq \
      tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=watchdog_builder /opt/watchdog_venv /opt/watchdog_venv

WORKDIR /watchdog
