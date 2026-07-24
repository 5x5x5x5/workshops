import modal

# No sandbox here (unlike the code example) — the math reward just parses the
# model's final answer and compares it to the gold answer, so there's no
# untrusted code to execute. That makes this pipeline simpler and a bit faster.
#
# All runtime deps are declared inline in the image, so the only local
# requirement is `modal`. Torch comes from the CUDA 12.4 wheels.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "trl>=0.15.0",
        "peft>=0.13.0",
        "datasets>=3.0.0",
        "accelerate>=0.34.0",
        "math-verify>=0.5.0",
    )
)

app = modal.App("grpo-math", image=image)

# Shared volume: the prepared dataset and the trained model both live here so
# each function can hand its artifacts to the next.
vol = modal.Volume.from_name("grpo-math", create_if_missing=True)
DATA_PATH = "/data"

# HuggingFace token secret — the models here are ungated, but attaching the
# token keeps things smooth. Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

# Resource profiles. A single T4 is enough for the 0.5B workshop model; the
# training code has an fp16 fallback for Turing GPUs (which lack bf16).
GPU = "T4"
GPU_MEMORY = 32768  # MB
CPU_MEMORY = 4096  # MB
