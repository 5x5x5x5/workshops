# RT-DETRv2 Object Detection

Fine-tune **RT-DETRv2** (real-time DETR, v2) on a custom COCO-format dataset, evaluate with COCO mAP, and deploy a live detection app — all on [Modal](https://modal.com).

## Why Object Detection Matters

Object detection is one of the most practical applications of deep learning. It powers autonomous vehicles, medical imaging, warehouse robotics, quality control on manufacturing lines, retail analytics, and more. Unlike image classification (which asks "what's in this image?"), object detection answers "what's in this image, where is it, and how many?"

Being able to fine-tune a detector on your own data — your products, your defects, your domain — and deploy it as a reliable service is a superpower for any team building real-world AI applications.

## Why DETR?

Traditional object detectors (Faster R-CNN, YOLO) rely on hand-designed components like anchor boxes and non-maximum suppression (NMS). **DETR** (DEtection TRansformer) replaced all of that with a transformer and a simple set-prediction loss. No anchors, no NMS, no post-processing hacks — just a clean end-to-end architecture.

**RT-DETRv2** takes this further with a hybrid CNN + transformer encoder for real-time speed while keeping the elegant DETR design. It matches or beats YOLO at similar speeds.

| | DETR | RT-DETR / RT-DETRv2 |
|---|---|---|
| End-to-end (no NMS) | yes | yes |
| Encoder | full transformer | hybrid (CNN + lightweight transformer) |
| Throughput | slow | real-time |
| Accuracy on COCO | baseline | matches or beats YOLO at similar speed |

The HuggingFace API is identical across DETR variants, so swapping between them is a one-line change.

## Why Modal?

ML pipelines are messy. Data prep, training, evaluation, and deployment each have different resource needs, failure modes, and iteration cycles. Modal gives you:

- **Resource isolation** — CPU for data prep, GPU for training/eval, lightweight containers for serving — declared per function
- **No infrastructure** — the container image (PyTorch, Transformers, CUDA) is defined inline in code; there's no cluster to provision
- **Shared volumes** — data prep hands the COCO split to training, and training hands the fine-tuned model to eval and to the server, via `modal.Volume`
- **Seamless deployment** — train a model, then serve it as a FastAPI endpoint + Gradio app with `modal deploy`
- **Scale** — run on your laptop or a fleet of GPUs with the same code

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Shared Modal image, app, and volumes — CPU for data prep, GPU for train/eval |
| `workflow.py` | Pipeline: prepare data → train → evaluate → inference demo |
| `app_server.py` | FastAPI model server — serves the fine-tuned model for inference |
| `app_gradio.py` | Gradio frontend — image upload + webcam detection UI |

## Setup

```bash
cd tutorials/detr-object-detection

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The only local dependency is `modal` — PyTorch, Transformers, and the rest are installed inside the Modal container image, defined in `config.py`.

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## HuggingFace secret

The pipeline pulls the dataset and base model from HuggingFace. The Modal functions read `HF_TOKEN` from a secret named `huggingface-secret`, so gated models and private datasets work anywhere the function runs. Create it once:

```bash
uv run modal secret create huggingface-secret HF_TOKEN=hf_...
```

(Or create it in the Modal dashboard under **Secrets** using the HuggingFace template.) The public defaults in this tutorial don't strictly need a token, but the secret must exist for the run to start — a token with read scope is fine.

## Run the Training Pipeline

### Default (RT-DETRv2-R18 on Union swag stickers)

```bash
uv run modal run workflow.py
```

### Quick test (smoke)

```bash
uv run modal run workflow.py --epochs 2 --batch-size 2 --demo-images 2
```

### With periodic mAP evaluation

Track mAP during training to catch overfitting or know when to stop:

```bash
uv run modal run workflow.py --epochs 50 --eval-every-n-epochs 10
```

### Swap model

```bash
# Larger RT-DETRv2 backbone (ResNet-50)
uv run modal run workflow.py --model-name "PekingU/rtdetr_v2_r50vd"

# RT-DETR v1 for comparison
uv run modal run workflow.py --model-name "PekingU/rtdetr_r18vd"

# Plain DETR (slower, original architecture)
uv run modal run workflow.py --model-name "facebook/detr-resnet-50"
```

### Swap dataset

The pipeline accepts any HF dataset with a COCO-format JSON and image directory:

```bash
uv run modal run workflow.py \
  --dataset-repo "your-org/your-coco-dataset" \
  --annotations-path "annotations/train.json" \
  --images-subdir "images"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `PekingU/rtdetr_v2_r18vd` | HuggingFace object-detection model |
| `--dataset-repo` | `sagecodes/union_flyte_swag_object_detection` | HF dataset repo id |
| `--annotations-path` | `swag/train.json` | Path to COCO JSON inside the repo |
| `--images-subdir` | `swag/images` | Path to image directory inside the repo |
| `--epochs` | `30` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch-size` | `4` | Per-device batch size |
| `--val-fraction` | `0.2` | Fraction of images held out for validation |
| `--threshold` | `0.5` | Score threshold for predictions in eval/demo |
| `--demo-images` | `8` | Number of val images rendered in the inference report |
| `--eval-every-n-epochs` | `None` | Run mAP eval every N epochs during training |

## Outputs & Reports

Modal has no built-in live-report UI, so `workflow.py` builds **styled HTML reports** and the `local_entrypoint` writes them to your working directory:

- `training_report.html` — training summary, loss chart, LR schedule, and periodic mAP (if `--eval-every-n-epochs` was set)
- `evaluation_report.html` — COCO mAP metrics table and bar chart with explanations of each metric
- `inference_demo.html` — side-by-side ground truth vs predictions with per-image mAP scores
- `pipeline_report.html` — final summary

Open them in a browser. The fine-tuned model is written to `./finetuned_model/` (and `finetuned_model.tar.gz`), and is also saved into the `rtdetr-model` volume so the detection server can load it.

Every run also streams logs to your terminal and is recorded in the [Modal dashboard](https://modal.com/apps) (`modal app list`), where you can inspect logs and container metrics per function.

## Deploy the Detection App

After training, deploy a live detection service: a FastAPI model server + Gradio web UI. Both load the fine-tuned model from the `rtdetr-model` volume written by the training pipeline.

### 1. The model server (`app_server.py`)

Loads the latest fine-tuned model from the volume once per container and exposes the detection API.

```bash
# Local dev with hot-reload (ephemeral URL, tears down on Ctrl-C)
uv run modal serve app_server.py

# Deploy a persistent URL
uv run modal deploy app_server.py
```

### 2. The Gradio frontend (`app_gradio.py`)

Auto-discovers the deployed server ("rtdetr-detection-server") and connects to it.

```bash
# Local dev with hot-reload
uv run modal serve app_gradio.py

# Deploy a persistent URL
uv run modal deploy app_gradio.py
```

`modal serve` prints an ephemeral web URL and hot-reloads on file changes — ideal while iterating. `modal deploy` publishes a persistent URL that stays up. To point the UI at a different endpoint, set `SERVER_URL`:

```bash
SERVER_URL=https://my-other-server.modal.run uv run modal serve app_gradio.py
```

### App API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/classes` | GET | List detected classes |
| `/detect` | POST | Upload image file, get bounding boxes |
| `/detect_base64` | POST | Send base64 image (webcam), get bounding boxes |
| `/docs` | GET | Interactive API docs (FastAPI auto-generated) |

## Data Augmentation

The training pipeline applies online augmentations via albumentations to increase data variety without expanding the dataset on disk:

- Horizontal/vertical flip
- Brightness, contrast, hue, and saturation jitter
- Small rotations (+/-15 degrees)
- Random scale (+/-20%)
- Gaussian blur and noise

All augmentations are bbox-aware — bounding box coordinates are automatically transformed to match the augmented image.

## Understanding the Metrics

- **mAP** (mean Average Precision) — the primary COCO metric. Averaged across IoU thresholds 0.50 to 0.95. Higher is better.
- **mAP@50** — mAP at a lenient 50% IoU overlap. Usually higher than mAP.
- **mAP@75** — mAP at a strict 75% IoU overlap. Tests precise box localization.
- **mAR@10** — mean Average Recall with up to 10 detections per image. Measures how many ground-truth objects the model finds.

## Choosing batch size

Inputs are resized to 640x640 by the HF image processor:

| GPU | VRAM | R18 | R50 (`rtdetr_v2_r50vd`) |
|---|---|---|---|
| T4 | 16 GB | **4** (default) | 2 |
| L4 / A10G | 24 GB | 8-16 | 4-8 |
| A100 | 40-80 GB | 32+ | 16 |

The pipeline defaults to an `A10G` GPU (set in `config.py` / the `@app.function` decorators). Change `gpu="A10G"` to `gpu="T4"` or `gpu="A100"` to trade cost for speed/VRAM.

## Notes on the default dataset

`sagecodes/union_flyte_swag_object_detection` has only ~18 images. It's enough to demo the full pipeline end-to-end, but swap in a larger dataset (`--dataset-repo`) for meaningful mAP numbers.
