"""
BERT Sentiment Classification — Fine-tune ModernBERT on IMDB reviews.

Fine-tune a ModernBERT (or any HuggingFace encoder) model on binary sentiment
classification. The pipeline downloads the IMDB dataset, trains the model,
and evaluates accuracy/F1 with example predictions.

Usage:
    # Quick local test
    uv run modal run workflow.py --max-train-samples 200 --max-eval-samples 50 --epochs 1

    # Default (ModernBERT-base on IMDB)
    uv run modal run workflow.py --epochs 3

    # Swap model
    uv run modal run workflow.py --model-name "bert-base-uncased"
"""

import json
import logging
import os

from config import DATA_PATH, app, data_volume, hf_secret

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Function 1: Prepare dataset
# ------------------------------------------------------------------

@app.function(cpu=2, memory=4096, timeout=1800, volumes={DATA_PATH: data_volume})
def prepare_data(
    dataset_name: str = "imdb",
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
) -> str:
    """Download IMDB dataset and save splits to the shared volume."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    train_ds = ds["train"].shuffle(seed=42).select(range(min(max_train_samples, len(ds["train"]))))
    eval_ds = ds["test"].shuffle(seed=42).select(range(min(max_eval_samples, len(ds["test"]))))

    processed = DatasetDict({"train": train_ds, "eval": eval_ds})

    output_dir = f"{DATA_PATH}/dataset"
    processed.save_to_disk(output_dir)
    data_volume.commit()
    log.info(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    return output_dir


# ------------------------------------------------------------------
# Function 2: Train
# ------------------------------------------------------------------

@app.function(
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={DATA_PATH: data_volume},
    secrets=[hf_secret],
)
def train(
    model_name: str,
    data_dir: str,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
) -> dict:
    """Fine-tune a BERT-style model for sentiment classification."""
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    hf_token = os.environ.get("HF_TOKEN")
    log.info(f"Training: model={model_name}")

    # -- Load data --
    data_volume.reload()
    dataset = load_from_disk(data_dir)

    # -- Tokenize --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # -- Load model --
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=hf_token,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

    # -- Collect training-loss history for the HTML report --
    train_history: list[dict] = []

    class HistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            train_history.append({
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
                "loss": round(logs["loss"], 4),
            })
            log.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"epoch {logs.get('epoch', 0):.1f} | loss {logs['loss']:.4f}"
            )

    # -- Train --
    output_dir = f"{DATA_PATH}/checkpoints"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_steps=50,
        report_to="none",
    )

    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[HistoryCallback()],
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete.")

    # -- Save --
    save_dir = f"{DATA_PATH}/finetuned_model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    data_volume.commit()
    log.info(f"Model saved to {save_dir}")

    metrics = trainer.evaluate()

    return {
        "model_path": save_dir,
        "train_history": train_history,
        "eval_accuracy": round(metrics.get("eval_accuracy", 0) * 100, 1),
        "eval_f1": round(metrics.get("eval_f1", 0) * 100, 1),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
    }


# ------------------------------------------------------------------
# Function 3: Evaluate — before/after comparison
# ------------------------------------------------------------------

@app.function(
    gpu="T4",
    memory=16384,
    timeout=1800,
    volumes={DATA_PATH: data_volume},
    secrets=[hf_secret],
)
def evaluate(
    model_name: str,
    finetuned_dir: str,
    data_dir: str,
    num_examples: int = 100,
) -> str:
    """Compare base model vs fine-tuned model on test examples."""
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    log.info("Starting evaluation...")

    # Load eval data (raw text + labels)
    data_volume.reload()
    dataset = load_from_disk(data_dir)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    texts = eval_ds["text"]
    labels = eval_ds["label"]

    def predict_batch(model, tokenizer, texts, batch_size=32):
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            preds.extend(batch_preds)
        return preds

    # -- Base model (untrained, random classifier head) --
    log.info(f"Loading base model: {model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=hf_token, num_labels=2,
    )
    base_model.eval()
    if torch.cuda.is_available():
        base_model = base_model.cuda()

    base_preds = predict_batch(base_model, base_tokenizer, texts)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Fine-tuned model --
    log.info("Loading fine-tuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
    ft_model = AutoModelForSequenceClassification.from_pretrained(finetuned_dir)
    ft_model.eval()
    if torch.cuda.is_available():
        ft_model = ft_model.cuda()

    ft_preds = predict_batch(ft_model, ft_tokenizer, texts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    label_names = {0: "negative", 1: "positive"}

    base_acc = accuracy_score(labels, base_preds) * 100
    base_f1 = f1_score(labels, base_preds, average="binary") * 100
    ft_acc = accuracy_score(labels, ft_preds) * 100
    ft_f1 = f1_score(labels, ft_preds, average="binary") * 100

    log.info(f"Base model — Accuracy: {base_acc:.1f}%, F1: {base_f1:.1f}%")
    log.info(f"Fine-tuned — Accuracy: {ft_acc:.1f}%, F1: {ft_f1:.1f}%")

    # -- Collect example comparisons for the report --
    comparisons = []
    for i in range(min(10, len(texts))):
        text_preview = texts[i][:300] + "..." if len(texts[i]) > 300 else texts[i]
        comparisons.append({
            "text": text_preview,
            "true_label": label_names[labels[i]],
            "base_pred": label_names[base_preds[i]],
            "ft_pred": label_names[ft_preds[i]],
            "base_correct": base_preds[i] == labels[i],
            "ft_correct": ft_preds[i] == labels[i],
        })

    return json.dumps({
        "base_accuracy": round(base_acc, 1),
        "base_f1": round(base_f1, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "finetuned_f1": round(ft_f1, 1),
        "accuracy_improvement": round(ft_acc - base_acc, 1),
        "num_examples": len(texts),
        "comparisons": comparisons,
    })


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@app.function()
def pipeline(
    model_name: str = "answerdotai/ModernBERT-base",
    dataset_name: str = "imdb",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 100,
) -> str:
    """
    End-to-end BERT fine-tuning pipeline for sentiment classification.

    1. Download and prepare IMDB dataset
    2. Fine-tune ModernBERT (or any encoder model) for binary classification
    3. Evaluate: before/after comparison on test set
    """
    log.info(f"Pipeline: {model_name} | dataset={dataset_name}")

    # Step 1: Prepare data
    data_dir = prepare_data.remote(dataset_name, max_train_samples, max_eval_samples)

    # Step 2: Train
    train_info = train.remote(model_name, data_dir, epochs, lr, batch_size)

    # Step 3: Evaluate
    result = evaluate.remote(model_name, train_info["model_path"], data_dir, num_eval_examples)
    metrics = json.loads(result)

    log.info(f"Pipeline complete. Improvement: {metrics['accuracy_improvement']:+.1f}pp")

    # Bundle everything the HTML report needs.
    return json.dumps({
        "model_name": model_name,
        "dataset_name": dataset_name,
        "train": train_info,
        "eval": metrics,
    })


# ------------------------------------------------------------------
# Report + local entrypoint
# ------------------------------------------------------------------

def build_report_html(payload: dict) -> str:
    """Render the same content the training/eval reports produced, as HTML."""
    model_name = payload["model_name"]
    dataset_name = payload["dataset_name"]
    train_info = payload["train"]
    ev = payload["eval"]

    # Training loss history rows.
    history_rows = "".join(
        f"<tr><td>{h['step']}</td><td>{h['epoch']:.1f}</td><td>{h['loss']:.4f}</td></tr>"
        for h in train_info["train_history"]
    )

    # Example predictions block.
    examples_html = ""
    for c in ev["comparisons"]:
        base_color = "green" if c["base_correct"] else "red"
        ft_color = "green" if c["ft_correct"] else "red"
        examples_html += f"""
<div style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:4px;">
<p style="font-size:0.9em;">{c['text']}</p>
<p><b>True:</b> {c['true_label']} |
<b>Base:</b> <span style="color:{base_color};">{c['base_pred']}</span> |
<b>Fine-tuned:</b> <span style="color:{ft_color};">{c['ft_pred']}</span></p>
</div>"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sentiment Classification Report</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;">
<h2>Sentiment Classification Pipeline</h2>
<p><b>Model:</b> {model_name}</p>
<p><b>Dataset:</b> {dataset_name}</p>

<h2>Training Complete — {model_name}</h2>
<p><b>Eval Accuracy:</b> {train_info['eval_accuracy']:.1f}%</p>
<p><b>Eval F1:</b> {train_info['eval_f1']:.1f}%</p>
<p><b>Epochs:</b> {train_info['epochs']} | <b>LR:</b> {train_info['lr']} | <b>Batch size:</b> {train_info['batch_size']}</p>
<h3>Training Loss History</h3>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Step</th><th>Epoch</th><th>Loss</th></tr>
{history_rows}
</table>

<hr/>
<h2>Evaluation Results</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th></th><th>Accuracy</th><th>F1</th></tr>
<tr><td><b>Base model</b></td><td>{ev['base_accuracy']:.1f}%</td><td>{ev['base_f1']:.1f}%</td></tr>
<tr><td><b>Fine-tuned</b></td><td>{ev['finetuned_accuracy']:.1f}%</td><td>{ev['finetuned_f1']:.1f}%</td></tr>
</table>
<p><b>Accuracy improvement:</b> {ev['accuracy_improvement']:+.1f} percentage points</p>
<hr/>
<h3>Example Predictions</h3>
{examples_html}
</body></html>"""


@app.local_entrypoint()
def main(
    model_name: str = "answerdotai/ModernBERT-base",
    dataset_name: str = "imdb",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 100,
):
    result = pipeline.remote(
        model_name,
        dataset_name,
        epochs,
        lr,
        batch_size,
        max_train_samples,
        max_eval_samples,
        num_eval_examples,
    )
    payload = json.loads(result)

    # Write the HTML report locally (Modal has no live report view).
    report_path = "bert_sentiment_report.html"
    with open(report_path, "w") as f:
        f.write(build_report_html(payload))

    ev = payload["eval"]
    print(f"Base accuracy:       {ev['base_accuracy']}%")
    print(f"Fine-tuned accuracy: {ev['finetuned_accuracy']}%")
    print(f"Improvement:         {ev['accuracy_improvement']:+.1f} percentage points")
    print(f"Wrote report to {report_path}")
