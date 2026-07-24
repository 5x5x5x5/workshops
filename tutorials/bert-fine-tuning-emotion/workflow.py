"""
ModernBERT Emotion Classification — Fine-tune, evaluate, and explore on Modal.

Fine-tune ModernBERT on emotion classification (6 classes: sadness, joy,
love, anger, fear, surprise) from Twitter text. The pipeline trains the model,
evaluates with confusion matrix and per-class metrics, then explores inference
with attention heatmaps and token importance visualization.

Each step builds a rich HTML report; because Modal has no live-report panel,
the reports are returned from the pipeline and the local entrypoint writes them
to disk as .html files you can open in a browser.

Usage:
    # Quick local-scale test (small dataset, one epoch)
    uv run modal run workflow.py --max-train-samples 200 --max-eval-samples 50 --epochs 1

    # Default (ModernBERT-base on emotion dataset)
    uv run modal run workflow.py

    # More data
    uv run modal run workflow.py --epochs 3 --max-train-samples 10000

    # Swap model
    uv run modal run workflow.py --model-name "bert-base-uncased"
"""

import io
import json
import logging
import os
import tarfile
from pathlib import Path

from config import (
    DATA_PATH,
    MODEL_PATH,
    app,
    data_volume,
    hf_secret,
    model_volume,
)
from report_helpers import (
    make_attention_text,
    make_bar_chart,
    make_confidence_bars,
    make_confusion_matrix,
    make_line_chart,
    make_token_importance_text,
    pipeline_step_indicator,
    wrap_report,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_DATASET = "dair-ai/emotion"

DATASET_DIR = f"{DATA_PATH}/dataset"
FINETUNED_DIR = f"{MODEL_PATH}/finetuned_model"


# ------------------------------------------------------------------
# Task 1: Get data
# ------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=4096,
    timeout=1800,
    volumes={DATA_PATH: data_volume},
    secrets=[hf_secret],
)
async def get_data(
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
) -> str:
    """Download the emotion dataset and save train/eval splits to the volume.

    The dair-ai/emotion dataset contains ~20k English Twitter messages labeled
    with one of 6 emotions: sadness, joy, love, anger, fear, surprise.
    """
    from datasets import DatasetDict, load_dataset

    log.info("Loading emotion dataset...")
    ds = load_dataset(EMOTION_DATASET)

    train_ds = ds["train"].shuffle(seed=42).select(range(min(max_train_samples, len(ds["train"]))))
    eval_ds = ds["test"].shuffle(seed=42).select(range(min(max_eval_samples, len(ds["test"]))))

    processed = DatasetDict({"train": train_ds, "eval": eval_ds})

    processed.save_to_disk(DATASET_DIR)
    data_volume.commit()
    log.info(f"Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval")

    return DATASET_DIR


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@app.function(
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={DATA_PATH: data_volume, MODEL_PATH: model_volume},
    secrets=[hf_secret],
)
async def train(
    model_name: str,
    data_dir: str,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
) -> tuple[str, str]:
    """Fine-tune a BERT-style model for 6-class emotion classification.

    Returns (report_html, finetuned_model_dir).
    """
    import numpy as np
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    hf_token = os.getenv("HF_TOKEN")
    log.info(f"Training: model={model_name}")

    id2label = {i: l for i, l in enumerate(EMOTION_LABELS)}
    label2id = {l: i for i, l in enumerate(EMOTION_LABELS)}

    # -- Load data --
    dataset = load_from_disk(data_dir)

    # -- Tokenize --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # -- Load model --
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=hf_token,
        num_labels=6,
        id2label=id2label,
        label2id=label2id,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # -- Metrics tracking for the report --
    training_log: list[dict] = []
    eval_log: list[dict] = []

    # -- Callback: collect loss/eval metrics for the charts --
    class ReportCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
            }
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 4)
            if "eval_accuracy" in logs:
                eval_log.append({
                    "epoch": entry["epoch"],
                    "accuracy": logs["eval_accuracy"],
                    "f1": logs.get("eval_f1", 0),
                    "eval_loss": logs.get("eval_loss", 0),
                })
            if "loss" in entry:
                training_log.append(entry)

    # -- Compute metrics --
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # -- Training --
    output_dir = "/tmp/checkpoints"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_steps=warmup_steps,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[ReportCallback()],
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete.")

    # -- Save model to the shared volume --
    trainer.save_model(FINETUNED_DIR)
    tokenizer.save_pretrained(FINETUNED_DIR)
    model_volume.commit()
    log.info(f"Model saved to {FINETUNED_DIR}")

    # -- Final eval + report --
    metrics = trainer.evaluate()
    final_acc = metrics.get("eval_accuracy", 0)
    final_f1 = metrics.get("eval_f1", 0)

    final_charts = ""
    loss_entries = [e for e in training_log if "loss" in e]
    if len(loss_entries) >= 2:
        loss_chart = make_line_chart(
            data=loss_entries,
            x_key="epoch",
            y_keys=["loss"],
            title="Training Loss",
            x_label="Epoch",
            y_label="Loss",
            colors=["#5a7db5"],
        )
        final_charts += f'<div class="chart-container">{loss_chart}</div>'

    if len(eval_log) >= 2:
        eval_chart = make_line_chart(
            data=eval_log,
            x_key="epoch",
            y_keys=["accuracy", "f1"],
            title="Eval Metrics Over Training",
            x_label="Epoch",
            y_label="Score",
            colors=["#0f3460", "#06d6a0"],
            y_max_cap=1.05,
            y_display_names={"accuracy": "Accuracy", "f1": "Weighted F1"},
        )
        final_charts += f'<div class="chart-container">{eval_chart}</div>'

    report_html = wrap_report(
        f"<h2>Training Complete</h2>"
        f"<h3>{model_name}</h3>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{final_acc:.1%}</div><div class="label">Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{final_f1:.1%}</div><div class="label">Weighted F1</div></div>'
        f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
        f'  <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>'
        f'</div>'
        f"{final_charts}"
    )

    return report_html, FINETUNED_DIR


# ------------------------------------------------------------------
# Task 3: Evaluate
# ------------------------------------------------------------------

@app.function(
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={DATA_PATH: data_volume, MODEL_PATH: model_volume},
    secrets=[hf_secret],
)
async def evaluate(
    model_name: str,
    finetuned_dir: str,
    data_dir: str,
    num_examples: int = 200,
) -> tuple[str, str]:
    """Compare base model (random head) vs fine-tuned on emotion classification.

    Produces confusion matrix, per-class precision/recall/F1, and overall metrics.
    Returns (metrics_json, report_html).
    """
    import torch
    from datasets import load_from_disk
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")
    log.info("Starting evaluation...")

    # -- Load eval data --
    dataset = load_from_disk(data_dir)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))
    texts = eval_ds["text"]
    labels = eval_ds["label"]

    def predict_batch(model, tokenizer, texts, batch_size=32):
        preds = []
        probs_all = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_probs = torch.softmax(outputs.logits, dim=-1).cpu()
            batch_preds = torch.argmax(batch_probs, dim=-1).tolist()
            preds.extend(batch_preds)
            probs_all.extend(batch_probs.tolist())
        return preds, probs_all

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=hf_token, num_labels=6,
    )
    base_model.eval()
    if torch.cuda.is_available():
        base_model = base_model.cuda()

    base_preds, base_probs = predict_batch(base_model, base_tokenizer, texts)
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

    ft_preds, ft_probs = predict_batch(ft_model, ft_tokenizer, texts)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Compute metrics --
    base_acc = accuracy_score(labels, base_preds) * 100
    base_f1 = f1_score(labels, base_preds, average="weighted") * 100
    ft_acc = accuracy_score(labels, ft_preds) * 100
    ft_f1 = f1_score(labels, ft_preds, average="weighted") * 100

    log.info(f"Base:      Accuracy={base_acc:.1f}%, F1={base_f1:.1f}%")
    log.info(f"Fine-tuned: Accuracy={ft_acc:.1f}%, F1={ft_f1:.1f}%")

    # -- Confusion matrix --
    ft_cm = sk_confusion_matrix(labels, ft_preds, labels=list(range(6)))
    cm_list = ft_cm.tolist()
    cm_svg = make_confusion_matrix(cm_list, EMOTION_LABELS, title="Fine-tuned Model — Confusion Matrix")

    # -- Per-class metrics --
    report_dict = classification_report(labels, ft_preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    per_class_html = "<table><tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"
    for label_name in EMOTION_LABELS:
        if label_name in report_dict:
            m = report_dict[label_name]
            per_class_html += (
                f"<tr><td><b>{label_name}</b></td>"
                f"<td>{m['precision']:.1%}</td>"
                f"<td>{m['recall']:.1%}</td>"
                f"<td>{m['f1-score']:.1%}</td>"
                f"<td>{int(m['support'])}</td></tr>"
            )
    per_class_html += "</table>"

    # -- Bar chart: base vs fine-tuned --
    per_class_base_acc = []
    per_class_ft_acc = []
    for cls_idx in range(6):
        cls_mask = [i for i, l in enumerate(labels) if l == cls_idx]
        if cls_mask:
            base_cls_acc = sum(1 for i in cls_mask if base_preds[i] == cls_idx) / len(cls_mask) * 100
            ft_cls_acc = sum(1 for i in cls_mask if ft_preds[i] == cls_idx) / len(cls_mask) * 100
        else:
            base_cls_acc = 0
            ft_cls_acc = 0
        per_class_base_acc.append(base_cls_acc)
        per_class_ft_acc.append(ft_cls_acc)

    bar_chart = make_bar_chart(
        labels=EMOTION_LABELS,
        series={"Base": per_class_base_acc, "Fine-tuned": per_class_ft_acc},
        title="Per-Class Accuracy — Base vs Fine-tuned",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=105.0,
    )

    # -- Example predictions --
    improvement = ft_acc - base_acc
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    examples_html = ""
    for i in range(min(10, len(texts))):
        true_label = EMOTION_LABELS[labels[i]]
        ft_label = EMOTION_LABELS[ft_preds[i]]
        base_label = EMOTION_LABELS[base_preds[i]]
        ft_correct = ft_preds[i] == labels[i]
        base_correct = base_preds[i] == labels[i]
        text_preview = texts[i][:200]

        ft_badge = "badge-success" if ft_correct else "badge-danger"
        base_badge = "badge-success" if base_correct else "badge-danger"

        examples_html += f"""
<div class="card">
  <p style="font-size:0.95em;">"{text_preview}"</p>
  <p>True: <b>{true_label}</b> |
  Base: <span class="badge {base_badge}">{base_label}</span> |
  Fine-tuned: <span class="badge {ft_badge}">{ft_label}</span></p>
</div>"""

    report_html = wrap_report(
        f"<h2>Evaluation Results — Emotion Classification</h2>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{base_acc:.1f}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{ft_acc:.1f}%</div><div class="label">Fine-tuned Accuracy</div></div>'
        f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
        f'  <div class="stat"><div class="value">{ft_f1:.1f}%</div><div class="label">Fine-tuned F1</div></div>'
        f'</div>'
        f'<div class="chart-container">{bar_chart}</div>'
        f'<div class="chart-container">{cm_svg}</div>'
        f"<h3>Per-Class Metrics (Fine-tuned)</h3>"
        f"{per_class_html}"
        f"<h3>Example Predictions</h3>"
        f"{examples_html}"
    )

    metrics_json = json.dumps({
        "base_accuracy": round(base_acc, 1),
        "base_f1": round(base_f1, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "finetuned_f1": round(ft_f1, 1),
        "improvement": round(improvement, 1),
        "num_examples": len(texts),
        "confusion_matrix": cm_list,
        "per_class": {k: report_dict[k] for k in EMOTION_LABELS if k in report_dict},
    })

    return metrics_json, report_html


# ------------------------------------------------------------------
# Task 4: Explore inference
# ------------------------------------------------------------------

@app.function(
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={DATA_PATH: data_volume, MODEL_PATH: model_volume},
    secrets=[hf_secret],
)
async def explore_inference(
    finetuned_dir: str,
    data_dir: str,
    num_examples: int = 8,
) -> tuple[str, str]:
    """Deep-dive into model behavior with attention and token importance.

    For a set of examples, this task produces:
    1. Predictions with full confidence distribution across all 6 emotions
    2. Attention heatmaps — which tokens the model focuses on for classification
       (CLS token attention from the last layer, averaged across heads)
    3. Token importance via gradient-based attribution — which tokens most
       influence the predicted class (gradient x embedding norm)
    4. Misclassification analysis — confident wrong predictions with explanations

    Returns (analysis_json, report_html).
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info("Starting explore_inference...")

    # -- Load model (with eager attention for weight extraction) --
    tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)

    # Need eager attention to extract attention weights (flash attention doesn't return them)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_dir,
        output_attentions=True,
        attn_implementation="eager",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -- Load eval data --
    dataset = load_from_disk(data_dir)
    eval_ds = dataset["eval"]

    # Pick a diverse set of examples — try to get some from each class
    examples_per_class = max(1, num_examples // 6)
    selected_indices = []
    for cls_idx in range(6):
        cls_indices = [i for i in range(len(eval_ds)) if eval_ds[i]["label"] == cls_idx]
        selected_indices.extend(cls_indices[:examples_per_class])
    # Fill remaining with random
    remaining = num_examples - len(selected_indices)
    if remaining > 0:
        other_indices = [i for i in range(len(eval_ds)) if i not in selected_indices]
        selected_indices.extend(other_indices[:remaining])
    selected_indices = selected_indices[:num_examples]

    # -- Analyze each example --
    analyses = []
    for idx_num, ds_idx in enumerate(selected_indices):
        text = eval_ds[ds_idx]["text"]
        true_label = eval_ds[ds_idx]["label"]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        token_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Forward pass with attention
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(logits).item())

        # -- Attention: CLS token attention from last layer --
        # attentions shape: (num_layers, batch, num_heads, seq_len, seq_len)
        last_layer_attention = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
        # Average across heads, take CLS row (index 0)
        cls_attention = last_layer_attention.mean(dim=0)[0].cpu().numpy()  # (seq_len,)

        # Remove [CLS] and [SEP] and padding from visualization
        real_token_mask = []
        clean_tokens = []
        clean_attention = []
        for i, tok in enumerate(tokens):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
                continue
            if tok == tokenizer.pad_token:
                continue
            clean_tokens.append(tok)
            clean_attention.append(float(cls_attention[i]))
            real_token_mask.append(i)

        # -- Token importance via gradient attribution --
        # Re-run with gradients enabled on embeddings
        embedding_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "word" in name.lower():
                embedding_layer = module
                break
        if embedding_layer is None:
            # Fallback: find the first large embedding
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Embedding) and module.weight.shape[0] > 1000:
                    embedding_layer = module
                    break

        importance_scores = [0.0] * len(clean_tokens)
        if embedding_layer is not None:
            inputs_grad = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs_grad = {k: v.to(device) for k, v in inputs_grad.items()}

            embeddings = embedding_layer(inputs_grad["input_ids"])
            embeddings.retain_grad()

            # Run model with embeddings instead of input_ids
            # We need to hook into the model to replace the embedding output
            embedding_output = [None]

            def hook_fn(module, input, output):
                embedding_output[0] = output
                return embeddings.requires_grad_(True)

            handle = embedding_layer.register_forward_hook(hook_fn)

            outputs_grad = model(**inputs_grad)
            handle.remove()

            # Gradient of predicted class w.r.t. embeddings
            pred_score = outputs_grad.logits[0, pred_idx]
            pred_score.backward()

            if embeddings.grad is not None:
                # Token importance = L2 norm of (gradient * embedding) per token
                token_importance = (embeddings.grad[0] * embeddings[0]).norm(dim=-1).detach().cpu().numpy()
                for clean_idx, orig_idx in enumerate(real_token_mask):
                    if orig_idx < len(token_importance):
                        importance_scores[clean_idx] = float(token_importance[orig_idx])

            model.zero_grad()

        analyses.append({
            "text": text,
            "true_label": true_label,
            "pred_idx": pred_idx,
            "probs": probs,
            "tokens": clean_tokens,
            "attention": clean_attention,
            "importance": importance_scores,
            "correct": pred_idx == true_label,
        })

    # -- Build report --
    log.info("Building explore_inference report...")

    # Overall summary
    correct = sum(1 for a in analyses if a["correct"])
    total = len(analyses)

    # Separate correct vs wrong
    wrong_analyses = [a for a in analyses if not a["correct"]]

    # -- Build example cards --
    examples_html = ""
    for a in analyses:
        true_name = EMOTION_LABELS[a["true_label"]]
        pred_name = EMOTION_LABELS[a["pred_idx"]]
        status_badge = "badge-success" if a["correct"] else "badge-danger"
        status_text = "Correct" if a["correct"] else "Wrong"

        # Confidence bars
        conf_bars = make_confidence_bars(
            labels=EMOTION_LABELS,
            probabilities=a["probs"],
            predicted_idx=a["pred_idx"],
            true_idx=a["true_label"],
        )

        # Attention heatmap
        attention_viz = make_attention_text(
            tokens=a["tokens"],
            weights=a["attention"],
            title="Attention (what the model looks at for its prediction — darker = more attention)",
        )

        # Token importance
        importance_viz = make_token_importance_text(
            tokens=a["tokens"],
            importance=a["importance"],
            title="Token importance (gradient attribution — green = supports prediction, red = opposes)",
        )

        text_preview = a["text"][:300]
        examples_html += f"""
<div class="card">
  <p style="font-size:1em;"><b>"{text_preview}"</b></p>
  <p>True: <b>{true_name}</b> | Predicted: <span class="badge {status_badge}">{pred_name} ({status_text})</span>
     | Confidence: <b>{a['probs'][a['pred_idx']]:.1%}</b></p>
  <div style="margin:12px 0;">{conf_bars}</div>
  <div style="margin:12px 0;">{attention_viz}</div>
  <div style="margin:12px 0;">{importance_viz}</div>
</div>"""

    # -- Misclassification spotlight --
    misclass_html = ""
    if wrong_analyses:
        # Sort by confidence (most confident wrong first)
        wrong_sorted = sorted(wrong_analyses, key=lambda a: a["probs"][a["pred_idx"]], reverse=True)

        misclass_html = "<h3>Misclassification Spotlight</h3>"
        misclass_html += '<div class="note">These are the model\'s most confident wrong predictions — cases where the model is sure but incorrect. These reveal the model\'s blind spots.</div>'

        for a in wrong_sorted[:3]:
            true_name = EMOTION_LABELS[a["true_label"]]
            pred_name = EMOTION_LABELS[a["pred_idx"]]
            conf = a["probs"][a["pred_idx"]]
            true_conf = a["probs"][a["true_label"]]

            misclass_html += f"""
<div class="card" style="border-left:4px solid #e63946;">
  <p><b>"{a['text'][:200]}"</b></p>
  <p>Predicted <span class="badge badge-danger">{pred_name}</span> ({conf:.1%})
     but true label is <span class="badge badge-info">{true_name}</span> ({true_conf:.1%})</p>
  <p style="font-size:0.85em;color:#6c757d;">
     The model assigned {conf:.1%} confidence to {pred_name} vs {true_conf:.1%} to {true_name}.
     {"The model was very sure here — this is a genuine blind spot." if conf > 0.7 else "The model was uncertain — the true class was a close second."}
  </p>
</div>"""

    report_html = wrap_report(
        f"<h2>Explore Inference — Attention &amp; Attribution</h2>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{correct}/{total}</div><div class="label">Correct</div></div>'
        f'  <div class="stat"><div class="value">{correct/total:.0%}</div><div class="label">Accuracy (sample)</div></div>'
        f'  <div class="stat"><div class="value">{len(wrong_analyses)}</div><div class="label">Errors to Analyze</div></div>'
        f'</div>'
        f'<div class="note">'
        f'<b>How to read the visualizations below:</b><br/>'
        f'<b>Attention heatmap:</b> Shows which tokens the [CLS] token attends to in the final layer '
        f'(averaged across all attention heads). Darker = more attention. This reveals what the model "looks at" when making its classification decision.<br/>'
        f'<b>Token importance:</b> Gradient-based attribution showing which tokens most influence the prediction. '
        f'Green = supports the prediction, Red = opposes it. Computed as gradient &times; embedding norm.'
        f'</div>'
        f"<h3>Example Analysis</h3>"
        f"{examples_html}"
        f"{misclass_html}"
    )

    analysis_json = json.dumps({
        "num_examples": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1),
        "num_misclassifications": len(wrong_analyses),
        "analyses": [
            {
                "text": a["text"][:200],
                "true_label": EMOTION_LABELS[a["true_label"]],
                "predicted": EMOTION_LABELS[a["pred_idx"]],
                "confidence": round(a["probs"][a["pred_idx"]], 3),
                "correct": a["correct"],
            }
            for a in analyses
        ],
    })

    return analysis_json, report_html


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@app.function(
    cpu=2,
    memory=4096,
    timeout=10800,
    volumes={DATA_PATH: data_volume, MODEL_PATH: model_volume},
    secrets=[hf_secret],
)
def pipeline(
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 200,
    num_explore_examples: int = 12,
) -> dict:
    """
    ModernBERT emotion classification pipeline.

    Returns a dict with the rendered HTML reports, the eval metrics, and a
    gzipped tarball of the fine-tuned model (also persisted to the model volume
    for serve.py). The local entrypoint writes these to disk.

    1. Download emotion dataset (6 classes from Twitter text)
    2. Fine-tune ModernBERT for sequence classification
    3. Evaluate: base vs fine-tuned with confusion matrix
    4. Explore inference: attention heatmaps + token importance
    """
    log.info(f"Pipeline: {model_name} | emotion classification")
    steps = ["Get Data", "Train", "Evaluate", "Explore Inference"]

    # Step 1: Get data
    data_dir = get_data.remote(max_train_samples, max_eval_samples)

    # Step 2: Train
    train_html, finetuned_dir = train.remote(
        model_name, data_dir, epochs, lr, batch_size, warmup_steps
    )

    # Step 3: Evaluate
    eval_result, eval_html = evaluate.remote(
        model_name, finetuned_dir, data_dir, num_eval_examples
    )
    eval_metrics = json.loads(eval_result)

    # Step 4: Explore inference
    explore_result, explore_html = explore_inference.remote(
        finetuned_dir, data_dir, num_explore_examples
    )

    # -- Final pipeline summary report --
    improvement = eval_metrics["improvement"]
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    pipeline_html = wrap_report(
        f"<h2>Emotion Classification Pipeline Complete</h2>"
        f"<h3>{model_name}</h3>"
        f"{pipeline_step_indicator(4, steps)}"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{eval_metrics["base_accuracy"]}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{eval_metrics["finetuned_accuracy"]}%</div><div class="label">Fine-tuned Accuracy</div></div>'
        f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
        f'  <div class="stat"><div class="value">{eval_metrics["finetuned_f1"]}%</div><div class="label">Weighted F1</div></div>'
        f'</div>'
    )

    # -- Tar the fine-tuned model so the entrypoint can save it locally --
    model_volume.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(finetuned_dir, arcname="finetuned_model")
    model_tar = buf.getvalue()

    log.info(f"Pipeline complete. Accuracy improvement: {improvement:+.1f}pp")

    return {
        "reports": {
            "pipeline": pipeline_html,
            "train": train_html,
            "evaluate": eval_html,
            "explore": explore_html,
        },
        "metrics": eval_metrics,
        "explore": json.loads(explore_result),
        "model_tar": model_tar,
        "model_dir": finetuned_dir,
    }


# ------------------------------------------------------------------
# Local entrypoint
# ------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    warmup_steps: int = 100,
    max_train_samples: int = 10000,
    max_eval_samples: int = 2000,
    num_eval_examples: int = 200,
    num_explore_examples: int = 12,
):
    """Run the pipeline in the cloud and write reports + model locally."""
    result = pipeline.remote(
        model_name,
        epochs,
        lr,
        batch_size,
        warmup_steps,
        max_train_samples,
        max_eval_samples,
        num_eval_examples,
        num_explore_examples,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Write the HTML reports (Modal has no live-report panel).
    for name, html in result["reports"].items():
        path = out_dir / f"{name}_report.html"
        path.write_text(html, encoding="utf-8")
        print(f"Wrote {path}")

    # Extract the fine-tuned model locally (also lives in the model volume).
    model_tar = result["model_tar"]
    with tarfile.open(fileobj=io.BytesIO(model_tar), mode="r:gz") as tar:
        tar.extractall(out_dir)
    print(f"Wrote {out_dir / 'finetuned_model'} ({len(model_tar):,} bytes tarball)")

    metrics = result["metrics"]
    print(
        f"Done. Base {metrics['base_accuracy']}% -> "
        f"Fine-tuned {metrics['finetuned_accuracy']}% "
        f"({metrics['improvement']:+.1f}pp), F1 {metrics['finetuned_f1']}%"
    )
