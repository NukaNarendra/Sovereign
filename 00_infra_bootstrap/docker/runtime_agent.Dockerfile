# ============================================================================
# SOVEREIGN PROBABILITY ENGINE — LIGHTWEIGHT SURVIVAL AGENT RUNTIME
# File: 00_infra_bootstrap/docker/runtime_agent.Dockerfile
# Runtime target: Python 3.11 / Debian Bookworm; Polygon Chain ID 137
# NOTE: This image does not contain ROCm or PyTorch training dependencies.
# ============================================================================

ARG PYTHON_BASE_IMAGE=python:3.11-slim-bookworm
ARG SOPS_VERSION=3.9.0
ARG AGE_VERSION=1.2.0

# ----------------------------------------------------------------------------
# Stage 1: Build Python wheels and fetch SOPS/Age binaries
# ----------------------------------------------------------------------------
FROM ${PYTHON_BASE_IMAGE} AS survival_wheel_builder

ARG SOPS_VERSION
ARG AGE_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CMAKE_ARGS="-DLLAMA_NATIVE=OFF -DLLAMA_AVX2=ON -DLLAMA_FMA=ON" \
    FORCE_CMAKE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ninja-build \
      curl \
      ca-certificates \
      git \
      libssl-dev \
      libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# These downloads target Linux amd64. For ARM64, use matching upstream binaries.
RUN set -eux; \
    curl -fsSL "https://github.com/FiloSottile/age/releases/download/v${AGE_VERSION}/age-v${AGE_VERSION}-linux-amd64.tar.gz" -o /tmp/age.tar.gz; \
    tar -xzf /tmp/age.tar.gz -C /tmp; \
    install -m 0755 /tmp/age/age /usr/local/bin/age; \
    install -m 0755 /tmp/age/age-keygen /usr/local/bin/age-keygen; \
    curl -fsSL "https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64" -o /usr/local/bin/sops; \
    chmod 0755 /usr/local/bin/sops; \
    rm -rf /tmp/age /tmp/age.tar.gz

RUN python3 -m venv /opt/sovereign_venv
ENV PATH="/opt/sovereign_venv/bin:${PATH}"

RUN pip install --upgrade "pip>=24.2" "setuptools>=72.1.0" "wheel>=0.44.0" && \
    pip install \
      py-clob-client==0.18.0 \
      web3==7.6.0 \
      eth-account==0.13.4 \
      eth-abi==5.1.0 \
      eth-utils==5.1.0 \
      hexbytes==1.2.1 \
      httpx==0.28.1 \
      aiohttp==3.11.11 \
      websockets==14.1 \
      tenacity==9.0.0 \
      ratelimit==2.2.1 \
      mcp==1.2.0 \
      fastmcp==0.4.1 \
      tavily-python==0.5.0 \
      fireworks-ai==0.15.12 \
      openai==1.59.3 \
      llama-cpp-python==0.3.4 \
      pydantic==2.10.4 \
      pydantic-settings==2.7.0 \
      orjson==3.10.12 \
      msgpack==1.1.0 \
      polars==1.18.0 \
      numpy==1.26.4 \
      rich==13.9.4 \
      typer==0.15.1 \
      loguru==0.7.3 \
      psutil==6.1.1 \
      prometheus-client==0.21.1

# Remove bytecode/cache to reduce image size (does not itself reduce runtime RSS).
RUN find /opt/sovereign_venv -type d -name '__pycache__' -prune -exec rm -rf '{}' + && \
    find /opt/sovereign_venv -type f -name '*.pyc' -delete

# ----------------------------------------------------------------------------
# Stage 2: Minimal production runtime
# ----------------------------------------------------------------------------
FROM ${PYTHON_BASE_IMAGE} AS survival_agent_runtime

LABEL org.opencontainers.image.title="Sovereign Probability Engine — Survival Trading Agent" \
      org.opencontainers.image.description="Lightweight Polygon CLOB runtime with MCP and fractional-Kelly configuration" \
      org.opencontainers.image.version="1.0.0-survival"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \