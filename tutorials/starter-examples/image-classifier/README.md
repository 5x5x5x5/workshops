# Image Classification Training

Fine-tune a pretrained ResNet18 on the Beans dataset from HuggingFace, on a GPU with [Modal](https://modal.com).

## What it does

- **`load_data`** — Downloads the Beans dataset (3 classes, ~1000 images), applies ImageNet transforms, saves tensors to a shared `modal.Volume`
- **`train`** — Fine-tunes ResNet18 with a replaced classification head on a T4 GPU, returns the trained weights
- **`pipeline`** — Orchestrates load data -> train
- **`main`** — A `local_entrypoint` that runs the pipeline and writes `resnet_beans.pt`

## Setup

```bash
cd tutorials/starter-examples/image-classifier

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## Run

```bash
uv run modal run image_classifier.py --num-epochs 3
```

The trained model is written to `resnet_beans.pt` in your working directory.

## Notes

- `load_data` runs on CPU and `train` runs on a T4 GPU — each `@app.function` requests exactly the resources it needs
- The dataset tensors are passed between functions through the `image-classifier-data` volume
- Dependencies are declared inline in the `modal.Image`, so the only local requirement is `modal`
