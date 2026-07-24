import modal

# All fine-tuning dependencies declared inline. CUDA-enabled Torch comes from the
# PyTorch wheel index, followed by the HuggingFace + PEFT/TRL/bitsandbytes stack.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets>=3.0.0",
        "bitsandbytes>=0.44.0",
        "accelerate>=0.34.0",
        "markdown",
    )
)

app = modal.App("llm-finetune-lora-qlora", image=image)

# Supplies HF_TOKEN as an env var inside functions that request it (gated models).
# Create it once with:  modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

# Shared volume so the prepared dataset and the fine-tuned model can be handed
# from one function to the next — and loaded later by the serving apps.
vol = modal.Volume.from_name("lora-qlora", create_if_missing=True)
DATA_PATH = "/data"
