import modal

# CUDA-enabled Torch plus the HuggingFace + sklearn stack, all declared inline.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "datasets>=3.0.0",
        "accelerate>=0.34.0",
        "scikit-learn",
    )
)

app = modal.App("bert-finetune-sentiment", image=image)

# Supplies HF_TOKEN as an env var inside functions that request it.
# Create it once with:  modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")

# Shared volume so prepared datasets and the fine-tuned model can be handed
# from one function to the next.
data_volume = modal.Volume.from_name("bert-sentiment-data", create_if_missing=True)
DATA_PATH = "/data"
