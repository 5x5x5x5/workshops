# ModernBERT Emotion Classification

Fine-tune ModernBERT to classify emotions in text, then explore *how* the model makes decisions with attention heatmaps and gradient-based token attribution — running on GPUs with [Modal](https://modal.com).

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/bert-fine-tuning-emotion/bert-emotion-tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

The dataset is [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — ~20k English Twitter messages labeled with one of 6 emotions:

| Label | Emotion | Example |
|-------|---------|---------|
| 0 | sadness | "i feel so empty inside" |
| 1 | joy | "i am so happy right now" |
| 2 | love | "i feel blessed to have you" |
| 3 | anger | "i am furious about this" |
| 4 | fear | "i feel so scared and anxious" |
| 5 | surprise | "i cant believe this just happened" |

## What's in the Pipeline

```
┌──────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────────┐
│ Get Data │───▶│   Train    │───▶│  Evaluate  │───▶│    Explore      │
│  (CPU)   │    │   (GPU)    │    │   (GPU)    │    │   Inference     │
└──────────┘    └────────────┘    └────────────┘    │    (GPU)        │
 emotion         ModernBERT        Confusion         └─────────────────┘
 dataset         fine-tuning       matrix +           Attention heatmaps
                 with loss/eval    per-class           + token importance
                 charts            metrics             + misclassification
                                                       analysis
```

1. **Get data** — Downloads the emotion dataset, shuffles, and splits into train/eval.
2. **Train** — Fine-tunes ModernBERT (or any HuggingFace encoder) for 6-class classification. The report shows loss curve, eval accuracy/F1, and final metrics.
3. **Evaluate** — Compares the base model (random classifier head) vs fine-tuned. Produces a confusion matrix heatmap, per-class precision/recall/F1, and a grouped bar chart of per-class accuracy.
4. **Explore inference** — The interesting part. For a set of examples, produces:
   - **Confidence distribution** — Softmax probabilities across all 6 emotions, not just the argmax
   - **Attention heatmap** — CLS token attention from the last transformer layer, averaged across heads. Shows which words the model "looks at" when classifying
   - **Token importance** — Gradient-based attribution (gradient x embedding norm) showing which tokens most influence the prediction. Green = supports prediction, red = opposes
   - **Misclassification spotlight** — The model's most confident wrong predictions, revealing blind spots

Each step builds a rich HTML report. Modal has no live-report panel, so the reports are returned from the pipeline and the local entrypoint writes them to `outputs/*.html` — open them in a browser after the run.

## Files

| File | What it does |
|------|-------------|
| `workflow.py` | Full pipeline — get_data, train, evaluate, explore_inference, and the `local_entrypoint` |
| `config.py` | Shared Modal image, app, HF secret, and volumes |
| `report_helpers.py` | SVG charts, confusion matrix, attention/importance visualization |
| `serve.py` | FastAPI model server on Modal — serves predictions with attention weights |
| `app_gradio.py` | Gradio frontend on Modal — interactive UI with attention heatmap |
| `requirements.txt` | Local dependency (just `modal`) |

All the heavy dependencies (torch, transformers, datasets, ...) are declared inline in the `modal.Image`, so the only thing you install locally is `modal`.

## Setup

```bash
cd tutorials/bert-fine-tuning-emotion
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

### HuggingFace token (optional)

ModernBERT doesn't require a token, but if you swap to a gated model you'll need one. Store it as a Modal secret named `huggingface-secret` (it's injected as `HF_TOKEN` into the containers):

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

The pipeline functions attach this secret. If you only use ungated models like ModernBERT, still create the secret (an empty/placeholder value is fine) so the functions can start.

## Run

### Quick test

```bash
uv run modal run workflow.py \
  --max-train-samples 200 \
  --max-eval-samples 50 \
  --epochs 1 \
  --num-eval-examples 30 \
  --num-explore-examples 6
```

Small dataset, one epoch — finishes quickly. Good for verifying the pipeline works end to end.

### Standard run

```bash
uv run modal run workflow.py \
  --model-name "answerdotai/ModernBERT-base" \
  --epochs 3 \
  --lr 2e-5 \
  --batch-size 16 \
  --max-train-samples 10000 \
  --max-eval-samples 2000 \
  --num-eval-examples 200 \
  --num-explore-examples 12
```

### With classic BERT

```bash
uv run modal run workflow.py --model-name "bert-base-uncased"
```

### Longer training

```bash
uv run modal run workflow.py \
  --epochs 5 \
  --max-train-samples 16000 \
  --num-eval-examples 500 \
  --num-explore-examples 18
```

After the run, the fine-tuned model is written to `outputs/finetuned_model/` and the reports to `outputs/*_report.html`. The model is also persisted to the `bert-emotion-model` Modal volume, which `serve.py` and `app_gradio.py` read from.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `answerdotai/ModernBERT-base` | HuggingFace encoder model to fine-tune |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch-size` | `16` | Batch size for training |
| `--warmup-steps` | `100` | Warmup steps for the scheduler |
| `--max-train-samples` | `10000` | Number of training examples |
| `--max-eval-samples` | `2000` | Number of held-out eval examples |
| `--num-eval-examples` | `200` | Examples used in the base vs fine-tuned comparison |
| `--num-explore-examples` | `12` | Examples for attention/attribution deep-dive |

## Inspect runs in the dashboard

Every `modal run` streams logs to your terminal and records the run in the [Modal dashboard](https://modal.com/apps), where you can browse past executions, logs, and per-function container metrics. List your apps with:

```bash
modal app list
```

or visit [modal.com/apps](https://modal.com/apps) directly.

## Model Serving

After training, deploy the model as a live API and a Gradio frontend. Both read the fine-tuned model from the `bert-emotion-model` volume, so run the pipeline first.

### The FastAPI server

```bash
# Dev server with hot-reload (ephemeral URL, tails logs)
uv run modal serve serve.py

# Persistent deployment (stable URL)
uv run modal deploy serve.py
```

This serves the fine-tuned model at a `/predict` endpoint that returns:
- Predicted emotion and confidence
- Full probability distribution across all 6 emotions
- Attention weights per token (for heatmap visualization)

The model loads once per container via `@modal.enter()` on an `@app.cls`, and the FastAPI app is exposed with `@modal.asgi_app()`.

Test the endpoint (use the URL printed by serve/deploy):

```bash
curl -X POST https://<your-app-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```

### The Gradio frontend

```bash
# Dev server with hot-reload
uv run modal serve app_gradio.py

# Persistent deployment
uv run modal deploy app_gradio.py
```

`modal serve` prints a URL you can open in the browser. The UI lets users:
- Type text and see emotion predictions with confidence bars
- View an attention heatmap showing which words the model focuses on
- Try pre-loaded example texts

The Gradio app loads the fine-tuned model directly from the shared volume and runs inference in-process, so it works on its own once training has produced a model. (`serve.py` is the standalone REST API if you'd rather split the UI from inference.)

## Why ModernBERT?

[ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) (2024) is a drop-in replacement for BERT with several improvements:

- **8192 token context** (vs BERT's 512) — handles longer text without truncation
- **Rotary positional embeddings** — better position encoding for variable-length inputs
- **Flash Attention** — faster training and inference
- **Better pretraining** — trained on more data with modern techniques
- **Same API** — works with `AutoModelForSequenceClassification` just like BERT

For this tutorial, the practical benefit is better classification accuracy with the same code. The attention visualization works the same way since it's still a multi-head attention transformer.

## Understanding the Visualizations

### Attention heatmap

The attention heatmap shows what the [CLS] token "looks at" in the final transformer layer. In BERT-style models, the [CLS] token is used for classification — its representation is fed to the classifier head. So the [CLS] attention pattern reveals which tokens the model considers most relevant for its emotion prediction.

For example, on "i am so happy right now":
- High attention on "happy" and "so" → the model correctly focuses on the emotional content
- Low attention on "i" and "right" → function words get less attention

### Token importance (gradient attribution)

This uses gradient-based attribution: for each token, we compute how much the token's embedding influences the predicted class score. Specifically:

```
importance(token) = ||gradient(prediction, embedding(token)) * embedding(token)||
```

Green tokens **support** the prediction, red tokens **oppose** it. This is complementary to attention — attention shows where the model looks, while attribution shows what actually drives the decision.

### Negation (a dataset gap)

Try "this does not make me angry". The model predicts **anger** with 99.8% confidence because it latches onto the word "angry" without properly handling "not" as a negation. The attention heatmap reveals this: "angry" gets high attention (0.44) and "not" gets some (0.41), but the model doesn't use the negation to flip the meaning.

This isn't a limitation of the model architecture. The training data (Twitter messages) consists mostly of direct emotional statements. Negated emotions like "I'm not sad anymore" or "this doesn't scare me" are rare in the dataset, so the model simply pattern-matches on emotional keywords. A dataset with more negated examples would teach the model to handle these correctly.

Try these in the Gradio app to explore the effect:
- "this does not make me angry" → predicts anger (wrong)
- "I used to be sad but now I'm fine" → may still predict sadness
- "I'm not surprised at all" → may still predict surprise

This is exactly why the attention visualization is valuable. You can *see* what the model latches onto and understand where the training data has gaps.

### Misclassification spotlight

The most confident wrong predictions are the most informative errors. A model that says "95% anger" on a text that's actually "fear" reveals something about the model's confusion boundary between those emotions. These often involve ambiguous text where emotions overlap (e.g., "i can't believe they did that" could be anger or surprise).
