"""Shared Modal image, app, secret, and volumes for the emotion pipeline."""

import modal

# Container image with all pipeline dependencies. Torch is installed from the
# CUDA 12.4 wheel index so training runs on the GPU.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "datasets>=3.0.0",
        "accelerate>=0.34.0",
        "scikit-learn",
        "numpy",
    )
)

app = modal.App("bert-emotion", image=image)

# HuggingFace token (needed only for gated models). Create it once with:
#   modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
# The secret injects HF_TOKEN into the container environment at run time.
hf_secret = modal.Secret.from_name("huggingface-secret")

# Shared volumes bridge the pipeline steps: `get_data` writes the dataset,
# `train` writes the fine-tuned model, and evaluate/explore/serve read them back.
data_volume = modal.Volume.from_name("bert-emotion-data", create_if_missing=True)
model_volume = modal.Volume.from_name("bert-emotion-model", create_if_missing=True)
DATA_PATH = "/data"
MODEL_PATH = "/models"
