# BERT Fine-Tuning: Sentiment Classification

Fine-tune ModernBERT (or any HuggingFace encoder) on IMDB movie reviews for binary sentiment classification, on a GPU with [Modal](https://modal.com). The pipeline trains the model and evaluates accuracy/F1 with a before/after comparison.

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Modal image, app, HuggingFace secret, and the shared data volume |
| `workflow.py` | Pipeline: prepare data → train → evaluate before/after |

## Setup

```bash
cd tutorials/bert-fine-tuning-sentiment

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The only local dependency is `modal` — all runtime deps (Torch, Transformers, Datasets, scikit-learn) are declared inline in the `modal.Image`.

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

### HuggingFace token (one-time)

The training and evaluation functions read `HF_TOKEN` from a Modal secret so gated
models can be downloaded. Create it once:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

(The default ModernBERT and IMDB datasets are public, but the secret is still required by the functions.)

## Run

### Default (ModernBERT on IMDB)

```bash
uv run modal run workflow.py --epochs 3
```

### Quick test

```bash
uv run modal run workflow.py --max-train-samples 200 --max-eval-samples 50 --epochs 1
```

### Swap model

```bash
# Classic BERT
uv run modal run workflow.py --model-name "bert-base-uncased"

# DistilBERT (smaller, faster)
uv run modal run workflow.py --model-name "distilbert-base-uncased"
```

`prepare_data` runs on CPU and `train`/`evaluate` run on a T4 GPU — each `@app.function`
requests exactly the resources it needs. The prepared dataset and fine-tuned model are
passed between functions through the `bert-sentiment-data` volume.

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `answerdotai/ModernBERT-base` | HuggingFace encoder model |
| `--dataset-name` | `imdb` | HuggingFace dataset |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-5` | Learning rate |
| `--batch-size` | `16` | Per-device batch size |
| `--max-train-samples` | `10000` | Max training examples |
| `--max-eval-samples` | `2000` | Max evaluation examples |
| `--num-eval-examples` | `100` | Examples for before/after comparison |

## Evaluation

The evaluate step runs the same test examples through both the base model (random classifier head) and the fine-tuned model, then compares:

- **Accuracy** and **F1 score**
- **Side-by-side predictions** showing base vs fine-tuned per review

The base model predicts essentially at random (~50%) since its classification head is untrained. The fine-tuned model should reach 85-90%+ accuracy.

## Report

When the run finishes, the `main` entrypoint writes an HTML report to
`bert_sentiment_report.html` in your working directory. It contains the training
loss history, final train metrics, the base-vs-fine-tuned results table, and a
sample of side-by-side example predictions.

```bash
uv run modal run workflow.py --max-train-samples 500 --max-eval-samples 100 --num-eval-examples 50 --epochs 5
```
