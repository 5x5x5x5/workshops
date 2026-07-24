"""Shared Modal image, app, and volumes for the RT-DETRv2 tutorial.

The image bundles PyTorch (CUDA 12.4 wheels), HuggingFace Transformers, and the
detection/eval stack. All dependencies are declared inline — the only thing you
need installed locally is `modal`.
"""

import modal

# All training/eval/inference deps live inside the container image.
#   - torch/torchvision from the CUDA 12.4 wheel index
#   - fonts-dejavu-core so PIL can draw readable bounding-box labels
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-dejavu-core")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.49.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.24.0",
        "datasets>=3.0.0",
        "pillow>=10.0.0",
        "albumentations>=1.4.0",
        "torchmetrics>=1.4.0",
        "pycocotools>=2.0.7",
        "numpy",
        "markdown",
    )
)

app = modal.App("rtdetr-detection", image=image)

# Shared volumes:
#   - data_volume  holds the prepared COCO split (images/ + train.json + val.json)
#     written by prepare_data and read by train/evaluate/inference_demo.
#   - model_volume holds the fine-tuned model, written by train and read by the
#     eval/demo tasks and by the detection server (app_server.py).
data_volume = modal.Volume.from_name("rtdetr-data", create_if_missing=True)
model_volume = modal.Volume.from_name("rtdetr-model", create_if_missing=True)

DATA_PATH = "/data"
MODEL_PATH = "/model"

# Where the fine-tuned model is saved inside model_volume.
FINETUNED_SUBDIR = "finetuned_model"
