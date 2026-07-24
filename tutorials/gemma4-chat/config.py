"""Shared config for the Gemma 4 chat app on Modal.

Switch between the MoE 26B-A4B and the dense 31B by flipping `MODEL`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import modal


@dataclass(frozen=True)
class ModelChoice:
    hf_repo: str
    model_id: str          # the name vLLM exposes over its OpenAI API
    app_name: str          # Modal app name (DNS-safe, lowercase)
    gpu: str               # Modal GPU spec, e.g. "A100" or "A100:2" for TP=2
    max_model_len: int


GEMMA_4_26B_A4B = ModelChoice(
    # `-it` (instruction-tuned). The base (non-it) model doesn't follow chat
    # format or activate thinking mode — it just continues text. The -it
    # repo also ships a chat_template.jinja so vLLM's /v1/chat/completions
    # works without us manually formatting prompts.
    hf_repo="google/gemma-4-26B-A4B-it",
    model_id="gemma-4-26b-a4b-it",
    app_name="gemma4-26b-a4b-it-vllm",
    # MoE: 26B total / 4B active, ~52 GB of safetensors. Fits comfortably on a
    # single 80 GB A100 alongside the KV cache.
    gpu="A100",
    max_model_len=8192,
)

GEMMA_4_31B = ModelChoice(
    hf_repo="google/gemma-4-31B-it",
    model_id="gemma-4-31b-it",
    app_name="gemma4-31b-it-vllm",
    # Dense bf16 ≈ 62 GB — tensor-parallel across 2 A100s so the weights plus
    # KV cache fit. `serve()` reads the `:2` count and passes
    # `--tensor-parallel-size 2` to vLLM.
    gpu="A100:2",
    max_model_len=8192,
)

# Pick which model to deploy. Override via env var to switch without code edits.
MODEL = GEMMA_4_31B if os.environ.get("GEMMA_VARIANT") == "31b" else GEMMA_4_26B_A4B

# Name of the Gradio frontend app.
CHAT_APP_NAME = "gemma4-chat-ui"

# HF_TOKEN — Gemma 4 weights are gated. Create the secret once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
# (the HF account must have accepted the Gemma license first).
hf_secret = modal.Secret.from_name("huggingface-secret")

# Cache the gated Gemma safetensors in a Volume so they download once and are
# reused by the vLLM server across cold starts.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"


def resolve_vllm_url() -> str:
    """Resolve the deployed vLLM server's base URL.

    The Gradio frontends call the vLLM OpenAI-compatible server over its public
    Modal web URL. Set `VLLM_URL` to override (e.g. point at a locally running
    `modal serve vllm_server.py` URL); otherwise we look up the deployed
    `serve` web endpoint of the vLLM app by name.
    """
    url = os.environ.get("VLLM_URL")
    if url:
        return url.rstrip("/")
    return modal.Function.from_name(MODEL.app_name, "serve").get_web_url().rstrip("/")
