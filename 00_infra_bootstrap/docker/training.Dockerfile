# ============================================================================
# SOVEREIGN PROBABILITY ENGINE — ROCm training runtime
# Target: AMD Instinct MI300X (gfx942), Linux x86_64
# File: 00_infra_bootstrap/docker/training.Dockerfile
#
# NOTE: ROCm/PyTorch/vLLM compatibility depends on the AMD host image and
# available wheels. Build and run the smoke tests below before long training.
# ============================================================================

ARG ROCM_BASE_IMAGE=rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0
ARG PYTORCH_ROCM_ARCH=gfx942
ARG SOPS_VERSION=3.9.0
ARG AGE_VERSION=1.2.0

# ---------------------------------------------------------------------------
# Stage 1: system foundation
# ---------------------------------------------------------------------------
FROM ${ROCM_BASE_IMAGE} AS rocm_system_foundation

ARG SOPS_VERSION
ARG AGE_VERSION

LABEL org.opencontainers.image.title="Sovereign Probability Engine — MI300X Trainer" \
      org.opencontainers.image.description="ROCm training runtime for CPT, SFT and GRPO experiments" \
      org.opencontainers.image.vendor="Sovereign Probability Engine" \
      org.opencontainers.image.version="1.0.0-rocm6.2"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    git-lfs \
    curl \
    wget \
    jq \
    tmux \
    htop \
    rsync \
    zstd \
    ca-certificates \
    gnupg \
    numactl \
    libnuma-dev \
    libaio-dev \
    libibverbs-dev \
    librdmacm-dev \
    openmpi-bin \
    libopenmpi-dev \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    coreutils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Mozilla SOPS and age binaries (Linux x86_64)
RUN set -eux; \
    curl -fsSL "https://github.com/FiloSottile/age/releases/download/v${AGE_VERSION}/age-v${AGE_VERSION}-linux-amd64.tar.gz" -o /tmp/age.tar.gz; \
    tar -xzf /tmp/age.tar.gz -C /tmp; \
    install -m 0755 /tmp/age/age /usr/local/bin/age; \
    install -m 0755 /tmp/age/age-keygen /usr/local/bin/age-keygen; \
    curl -fsSL "https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64" -o /usr/local/bin/sops; \
    chmod 0755 /usr/local/bin/sops; \
    rm -rf /tmp/age /tmp/age.tar.gz

# ---------------------------------------------------------------------------
# Stage 2: Python / ML dependencies
# ---------------------------------------------------------------------------
FROM rocm_system_foundation AS rocm_python_stack

ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    HSA_OVERRIDE_GFX_VERSION=9.4.2 \
    DS_BUILD_AIO=1 \
    DS_BUILD_SPARSE_ATTN=0 \
    DS_BUILD_EVOFORMER_ATTN=0

RUN python3 -m pip install --upgrade \
    'pip>=24.2' \
    'setuptools>=72.1.0' \
    'wheel>=0.44.0'

# Use the ROCm 6.2 wheel index. The base image already supplies ROCm libraries.
# If these exact wheels are unavailable for the selected image, align the
# PyTorch versions with the ROCm version supported by that image.
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/rocm6.2 \
    'torch==2.5.1' \