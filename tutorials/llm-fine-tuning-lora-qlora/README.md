# LLM Fine-Tuning: Text-to-SQL

Fine-tune a language model on text-to-SQL with full fine-tuning, LoRA, or QLoRA — all in one [Modal](https://modal.com) pipeline. Then serve the result as a FastAPI endpoint with a Gradio UI.

**Default model:** [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — a tiny 135M parameter model. Small enough to train quickly on a single GPU, large enough to learn the SQL pattern and demonstrate the difference between fine-tuning methods.

<a target="_blank" href="https://colab.research.google.com/github/unionai/workshops/blob/main/tutorials/llm-fine-tuning-lora-qlora/llm-fine-tune-tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Modal app, image, HuggingFace secret, and shared volume |
| `workflow.py` | Pipeline: prepare data → train (full/LoRA/QLoRA) → evaluate before/after |
| `report_helpers.py` | Report CSS, SVG chart generators (line/bar), and HTML helpers |
| `serve.py` | Serve the fine-tuned model as a FastAPI endpoint |
| `app_gradio.py` | Gradio UI for interactive text-to-SQL queries |

## Setup

```bash
cd tutorials/llm-fine-tuning-lora-qlora

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## HuggingFace secret

The training and serving functions read `HF_TOKEN` from a Modal secret named
`huggingface-secret` (needed for gated models; optional otherwise). Create it once:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

(Or create it in the Modal dashboard under **Secrets**.)

## How LoRA Works

**Full fine-tuning** updates every weight in the model — effective but expensive. For a 7B model that's billions of parameters to train, store, and deploy.

**LoRA (Low-Rank Adaptation)** takes a different approach: freeze the entire base model and inject small trainable adapters alongside the original weights. Instead of modifying a large weight matrix `W` directly, LoRA adds a low-rank decomposition `A × B` that learns a small correction:

```
                    ┌─────────────────────────┐
                    │   Original Weight W      │
input ────────────→ │   (frozen, e.g. 768×768) │──→ main output
    │               └─────────────────────────┘         │
    │               ┌───────────┐ ┌───────────┐         │
    └─────────────→ │ A (768×16) │→│ B (16×768) │→ × α/r ──→ + ──→ combined output
                    └───────────┘ └───────────┘
                    (LoRA adapter, trainable)
```

The original weight `W` stays completely frozen. The adapter matrices `A` and `B` are tiny — for a 768×768 layer with rank `r=16`, LoRA adds only 24,576 params vs the original 589,824 (~4%).

**Key parameters:**
- **`r` (rank)** — size of the adapter matrices. Higher = more capacity but more params
- **`alpha`** — scaling factor. The adapter output is multiplied by `alpha/r` before being added. Controls how strongly the adapter influences the output. Common practice: `alpha = 2 × r`
- **`alpha` adds zero extra parameters** — it's just a scalar multiplier

These small corrections are applied at every key layer in every transformer block — which is enough to steer the model's behavior significantly while training only 2-4% of total parameters.

**QLoRA** goes further: it quantizes the frozen base model to 4-bit precision, reducing memory even more. The LoRA adapters still train in full precision. This lets you fine-tune models that wouldn't otherwise fit in GPU memory.

## Fine-Tuning Methods

| Method | What happens | Memory | Best for |
|--------|-------------|--------|----------|
| `full` | Train all model parameters | High | Small models, maximum quality |
| `lora` | Freeze base, train low-rank adapters | Medium | Good balance of quality and efficiency |
| `qlora` | 4-bit quantized base + LoRA adapters | Low | Larger models on limited GPU memory |

> **Note on QLoRA:** With a small model like SmolLM2-135M, QLoRA is overkill — the model already fits easily in GPU memory, and 4-bit quantization just hurts quality. QLoRA shines when you need to fine-tune a model that's too large to fit in VRAM otherwise (e.g., a 7B+ model on a single T4). It's included here to show *how* it works so you can apply it when you need it.

## Run

Each `modal run` executes the whole pipeline in the cloud on GPU. The
fine-tuned model is written to the shared `lora-qlora` volume, and the
training / evaluation / pipeline HTML reports are saved to your working
directory (`training_report.html`, `evaluation_report.html`, `report.html`).

### LoRA (default)

```bash
modal run workflow.py --method lora
```

### QLoRA

```bash
modal run workflow.py --method qlora
```

### Full fine-tuning

```bash
modal run workflow.py --method full
```

### Quick test (small subset)

```bash
modal run workflow.py \
  --max-train-samples 100 --max-eval-samples 20 --epochs 3
```

### Swap model

```bash
modal run workflow.py \
  --method lora --model-name "Qwen/Qwen2.5-0.5B" --epochs 3
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model to fine-tune |
| `--dataset-name` | `b-mc2/sql-create-context` | HuggingFace dataset |
| `--method` | `lora` | Fine-tuning method: `full`, `lora`, or `qlora` |
| `--epochs` | `3` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--batch-size` | `4` | Per-device batch size |
| `--max-train-samples` | `5000` | Max training examples |
| `--max-eval-samples` | `500` | Max evaluation examples |
| `--num-eval-examples` | `50` | Examples for before/after comparison |
| `--lora-r` | `16` | LoRA rank (for lora/qlora) |
| `--lora-alpha` | `32` | LoRA alpha (for lora/qlora) |

## Evaluation

The evaluate step runs the same prompts through both the base model and the fine-tuned model, then compares:

- **Exact match accuracy** on generated SQL
- **Side-by-side examples** showing base vs fine-tuned output
- **Improvement** in percentage points

The report shows the **full raw output** from each model. This is intentional — one of the clearest effects of fine-tuning is that the base model tends to ramble (repeating the prompt template, generating extra text after the SQL), while the fine-tuned model learns to stop cleanly after the answer thanks to the EOS token in the training data.

For scoring, `normalize_sql` extracts just the first SQL statement (truncating at `###` or newline) so the accuracy comparison is fair even when the base model keeps generating.

Results are written as self-contained HTML reports (`training_report.html`,
`evaluation_report.html`, `report.html`) with stat grids, SVG charts, and
side-by-side comparisons — open them in any browser. Live run logs and status
are visible in the [Modal dashboard](https://modal.com/apps) or via `modal app list`.

## Serve the Fine-Tuned Model

After training, the model sits on the `lora-qlora` volume. Serve it as a FastAPI endpoint:

```bash
# Dev server with a live-reloading URL
modal serve serve.py

# Deploy a persistent endpoint
modal deploy serve.py
```

`serve.py` loads the model once per container (via `@modal.enter`) from the
volume and serves SQL generation. The endpoint URLs are printed when you serve
or deploy.

Test the endpoint:

```bash
curl -X POST https://<your-workspace>--finetuned-sql-api-model-generate.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
    "question": "What is the average salary by department?"
  }'
```

Response:

```json
{
  "sql": "SELECT department, AVG(salary) FROM employees GROUP BY department",
  "raw_output": "SELECT department, AVG(salary) FROM employees GROUP BY department"
}
```

## Gradio UI

Deploy an interactive frontend for the model. It loads the fine-tuned model
directly from the volume — no separate server required:

```bash
# Dev server with a live-reloading URL
modal serve app_gradio.py

# Deploy a persistent URL
modal deploy app_gradio.py
```

Includes example schemas and questions to try out.

## Swapping Models and Datasets

Everything is HuggingFace-based, so swapping is just changing a string:

```bash
# Different model
modal run workflow.py --model-name "Qwen/Qwen2.5-0.5B"

# Different dataset (must have similar structure or update format_example in workflow.py)
modal run workflow.py --dataset-name "your-org/your-dataset"
```

### LoRA target modules

When swapping models, be aware that **LoRA target module names vary between architectures**. The default targets in `workflow.py` are set for LLaMA-style models (SmolLM2, Qwen, Mistral, etc.):

```python
target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

These are the layers inside each transformer block where LoRA injects low-rank adapters:

**Attention layers** — how the model decides what to focus on:
| Module | Name | What it does |
|--------|------|-------------|
| `q_proj` | Query | What to look for in the context |
| `k_proj` | Key | What each token offers to match against |
| `v_proj` | Value | What information to extract once matched |
| `o_proj` | Output | Combines multi-head attention results |

**MLP (feed-forward) layers** — how the model processes information after attention:
| Module | Name | What it does |
|--------|------|-------------|
| `gate_proj` | Gate | Controls how much information flows through (SwiGLU activation) |
| `up_proj` | Up | Projects to a higher dimension for richer representations |
| `down_proj` | Down | Projects back down to the model's hidden size |

By targeting all seven layers, LoRA can adapt both *what the model pays attention to* and *how it processes that information* — without retraining all the weights.

Other architectures use different naming conventions:

| Architecture | Attention modules | Example models |
|-------------|------------------|----------------|
| LLaMA-style | `q_proj`, `k_proj`, `v_proj`, `o_proj` | SmolLM2, Qwen, Mistral, LLaMA |
| GPT-2 / GPT-Neo | `q_proj`, `k_proj`, `v_proj`, `out_proj` | GPT-2, GPT-Neo, GPT-J |
| BLOOM | `query_key_value`, `dense` | BLOOM, BLOOMZ |
| Falcon | `query_key_value`, `dense` | Falcon |
| Phi | `q_proj`, `k_proj`, `v_proj`, `dense` | Phi-1, Phi-2 |

If you see LoRA training with 0 trainable parameters or an error about missing modules, check the model's attention layer names:

```python
# Quick way to find the right module names
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-model")
print([n for n, _ in model.named_modules() if "proj" in n or "query" in n or "dense" in n])
```

### Good models to try

| Model | Params | Notes |
|-------|--------|-------|
| `HuggingFaceTB/SmolLM2-135M` | 135M | Default — fast training, good for demos |
| `Qwen/Qwen2.5-0.5B` | 500M | Better quality, still fits easily on a T4 |
| `Qwen/Qwen2.5-1.5B` | 1.5B | Good quality, may benefit from QLoRA on smaller GPUs |
| `meta-llama/Llama-3.2-1B` | 1B | Strong base model, requires HF token (gated) |
