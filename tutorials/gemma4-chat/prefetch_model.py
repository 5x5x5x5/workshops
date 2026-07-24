"""Prefetch a Gemma 4 model from Hugging Face into the Modal `hf-cache` Volume.

Standalone helper — useful when you want to download/cache the gated weights
before deploying, or to verify the model + `huggingface-secret` work. The vLLM
server also downloads on first cold start (into the same Volume), so running
this first just moves that one-time download out of the serving path.

Usage:
    modal secret create huggingface-secret HF_TOKEN=hf_...   # one-time
    modal run prefetch_model.py            # 26B-A4B by default
    GEMMA_VARIANT=31b modal run prefetch_model.py
"""

from __future__ import annotations

import modal

from config import MODEL, hf_secret, hf_cache, HF_CACHE_PATH

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_CACHE_PATH})
)

app = modal.App("gemma4-prefetch", image=image)


@app.function(
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=3600,
)
def prefetch() -> None:
    from huggingface_hub import snapshot_download

    print(f"Prefetching {MODEL.hf_repo} → hf-cache Volume…")
    # HF_TOKEN comes from the huggingface-secret; the account must have accepted
    # the Gemma license.
    snapshot_download(MODEL.hf_repo)
    hf_cache.commit()
    print(f"Done. {MODEL.hf_repo} is cached and ready for vllm_server.py.")


@app.local_entrypoint()
def main():
    prefetch.remote()
