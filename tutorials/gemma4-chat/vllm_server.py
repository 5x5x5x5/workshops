"""vLLM model-serving app for Gemma 4 on Modal.

Runs the vLLM OpenAI-compatible server as a Modal web endpoint. The frontends
(`chat_app.py`, `vision_app.py`, `live_camera_app.py`) resolve this server's
public URL by name and talk to it over `/v1/chat/completions`, `/docs`, etc.

Dev (hot-reload, ephemeral URL):
    modal serve vllm_server.py

Deploy (persistent URL):
    modal deploy vllm_server.py
    # or, for the dense 31B variant:
    GEMMA_VARIANT=31b modal deploy vllm_server.py
"""

from __future__ import annotations

import modal

from config import MODEL, hf_secret, hf_cache, HF_CACHE_PATH


# vLLM image. We start from NVIDIA's prebuilt Gemma 4 vLLM container — Gemma 4's
# architecture (`Gemma4ForConditionalGeneration`) is too new for vanilla vLLM,
# and the `gemma4-cu130` tag bundles the matching vLLM patches + CUDA runtime.
# `.entrypoint([])` clears the image's default vLLM entrypoint so Modal can run
# our `serve` function instead.
vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:gemma4-cu130")
    .entrypoint([])
    # Let HuggingFace resolve its cache from the mounted Volume.
    .env({"HF_HOME": HF_CACHE_PATH})
)

app = modal.App(MODEL.app_name, image=vllm_image)


@app.function(
    gpu=MODEL.gpu,
    secrets=[hf_secret],
    volumes={HF_CACHE_PATH: hf_cache},
    timeout=3600,
    # Scale to zero when idle. Cold starts take a few minutes (image pull +
    # weight load + kernel compile), so amortize warm-up over a generous window.
    scaledown_window=1800,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Launch the vLLM OpenAI-compatible server on port 8000."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL.hf_repo,
        # The name vLLM exposes over its OpenAI API (what the frontends pass as
        # `model=`).
        "--served-model-name",
        MODEL.model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        str(MODEL.max_model_len),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.90",
    ]

    # Typed GPU with a count (e.g. "A100:2") → tensor-parallel across that many.
    if ":" in MODEL.gpu:
        cmd += ["--tensor-parallel-size", MODEL.gpu.split(":")[1]]

    subprocess.Popen(" ".join(cmd), shell=True)
