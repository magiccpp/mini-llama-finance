FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# ── system packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        vim \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements-train.txt ./
RUN pip install -r requirements-train.txt

# ── HuggingFace cache ────────────────────────────────────────────────────────
ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── project source ───────────────────────────────────────────────────────────
COPY *.py ./
COPY processors/ ./processors/
COPY scrapers/   ./scrapers/

RUN mkdir -p /workspace/data \
             /workspace/checkpoints \
             /workspace/logs \
             /hf_cache

CMD ["/bin/bash"]
