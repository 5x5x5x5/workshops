"""
LLM Fine-Tuning Pipeline — Full, LoRA, and QLoRA in one workflow.

One pipeline, three fine-tuning methods. The `method` flag switches between:
  - full:  train all parameters (best quality, most compute)
  - lora:  freeze base model, train low-rank adapters (good balance)
  - qlora: 4-bit quantized base + LoRA adapters (least memory)

Usage:
    # LoRA fine-tuning (default)
    modal run workflow.py

    # QLoRA
    modal run workflow.py --method qlora

    # Full fine-tuning with a small model
    modal run workflow.py --method full

    # Quick test (few samples)
    modal run workflow.py --max-train-samples 100 --max-eval-samples 20 --epochs 1
"""

import json
import logging

from config import DATA_PATH, app, hf_secret, vol
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATASET_DIR = f"{DATA_PATH}/dataset"
MODEL_DIR = f"{DATA_PATH}/finetuned_model"


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@app.function(cpu=2, memory=4096, volumes={DATA_PATH: vol}, timeout=3600)
def prepare_data(
    dataset_name: str = "b-mc2/sql-create-context",
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
) -> str:
    """Download dataset from HuggingFace and format for instruction fine-tuning."""
    from datasets import DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    def format_example(ex):
        return {
            "text": (
                "### Task: Generate a SQL query to answer the question.\n"
                f"### Schema:\n{ex['context']}\n"
                f"### Question:\n{ex['question']}\n"
                f"### SQL:\n{ex['answer']}\n<|endoftext|>"
            )
        }

    ds = ds.map(format_example)

    # Split into train and eval
    total = len(ds)
    train_end = min(max_train_samples, total - max_eval_samples)
    eval_start = train_end
    eval_end = min(eval_start + max_eval_samples, total)

    processed = DatasetDict({
        "train": ds.select(range(train_end)),
        "eval": ds.select(range(eval_start, eval_end)),
    })

    processed.save_to_disk(DATASET_DIR)
    vol.commit()
    log.info(f"Dataset ready: {len(processed['train'])} train, {len(processed['eval'])} eval")

    return DATASET_DIR


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@app.function(
    gpu="A100",
    cpu=4,
    memory=24576,
    secrets=[hf_secret],
    volumes={DATA_PATH: vol},
    timeout=7200,
)
def train(
    model_name: str,
    data_dir: str,
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict:
    """Fine-tune a model using full, LoRA, or QLoRA method.

    Returns the volume path of the saved model plus a self-contained HTML
    training report (loss / learning-rate charts) for the caller to write out.
    """
    import os

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    log.info(f"Training: model={model_name}, method={method}")

    HF_TOKEN = os.environ.get("HF_TOKEN")

    # -- Load data --
    vol.reload()
    dataset = load_from_disk(data_dir)

    # -- Load tokenizer --
    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if method == "qlora":
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **token_kwargs,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **token_kwargs,
            dtype=dtype,
            device_map="auto",
        )

    # -- Apply LoRA adapters --
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=lora_r,              # Rank — size of the low-rank matrices. Higher = more capacity but more params
            lora_alpha=lora_alpha,  # Scaling factor — controls adapter impact. Effective scale = alpha/r
            # Attention layers — LoRA adapters inject low-rank updates here:
            #   q_proj (Query)     — what to look for in context
            #   k_proj (Key)       — what each token offers to match against
            #   v_proj (Value)     — what information to extract once matched
            #   o_proj (Output)    — combines multi-head attention results
            # MLP layers — LoRA adapters also update the feed-forward network:
            #   gate_proj (Gate)   — controls how much information flows through (SwiGLU activation)
            #   up_proj (Up)       — projects to a higher dimension for richer representations
            #   down_proj (Down)   — projects back down to the model's hidden size
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,     # Dropout on adapter weights — light regularization to prevent overfitting
            bias="none",           # Don't train bias terms — keeps adapter small and stable
            task_type="CAUSAL_LM", # Tells PEFT this is a text generation model (vs classification, etc.)
        )
        model = get_peft_model(model, lora_config)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        log.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params / total_params * 100:.1f}%)")

    # -- Training log state (populated by the callback below) --
    training_log: list[dict] = []

    method_badge = f'<span class="badge badge-info">{method.upper()}</span>'
    if method == "qlora":
        method_badge = f'<span class="badge badge-success">QLoRA (4-bit)</span>'
    elif method == "full":
        method_badge = f'<span class="badge badge-danger">Full Fine-Tune</span>'

    # -- Metrics callback: accumulate per-step training metrics --
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
                "loss": round(logs["loss"], 4),
            }
            if "learning_rate" in logs:
                entry["lr"] = logs["learning_rate"]
            if "grad_norm" in logs:
                entry["grad_norm"] = round(float(logs["grad_norm"]), 4)
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f} "
                f"loss={entry['loss']:.4f}"
            )

    # -- Train --
    output_dir = "/tmp/checkpoints"
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    log.info("Starting training...")
    trainer.train()
    log.info("Training complete.")

    # -- Merge LoRA weights and save to the shared volume --
    if method in ("lora", "qlora"):
        log.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    vol.commit()
    log.info(f"Model saved to {MODEL_DIR}")

    # -- Final training report --
    final_loss = training_log[-1]["loss"] if training_log else "N/A"

    loss_chart = make_line_chart(
        data=training_log,
        x_key="epoch",
        y_keys=["loss"],
        title="Training Loss",
        x_label="Epoch",
        y_label="Loss",
        colors=["#5a7db5"],
    ) if training_log else ""

    lr_chart = ""
    if training_log and "lr" in training_log[0]:
        lr_chart = make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["lr"],
            title="Learning Rate Schedule",
            x_label="Epoch",
            y_label="LR",
            colors=["#0f3460"],
        )

    report_html = wrap_report(
        f"<h2>Training Complete</h2>"
        f"<h3>{model_name}</h3>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{method.upper()}</div><div class="label">Method</div></div>'
        f'  <div class="stat"><div class="value">{final_loss}</div><div class="label">Final Loss</div></div>'
        f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
        f'  <div class="stat"><div class="value">{total_params:,}</div><div class="label">Total Params</div></div>'
        f'  <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Trainable Params</div></div>'
        f'  <div class="stat"><div class="value">{trainable_params / total_params * 100:.1f}%</div><div class="label">% Trainable</div></div>'
        f'</div>'
        f'<p>Method: {method_badge} | Total params: {total_params:,} | Trainable: {trainable_params:,}</p>'
        f'<div class="chart-container">{loss_chart}</div>'
        f'{f"""<div class="chart-container">{lr_chart}</div>""" if lr_chart else ""}'
    )

    return {"model_dir": MODEL_DIR, "report_html": report_html}


# ------------------------------------------------------------------
# Task 3: Evaluate — before/after comparison
# ------------------------------------------------------------------

@app.function(
    gpu="A100",
    cpu=4,
    memory=24576,
    secrets=[hf_secret],
    volumes={DATA_PATH: vol},
    timeout=3600,
)
def evaluate(
    model_name: str,
    finetuned_dir: str,
    data_dir: str,
    num_examples: int = 50,
) -> dict:
    """Compare base model vs fine-tuned model on test examples.

    Returns the metrics dict plus a self-contained HTML evaluation report.
    """
    import os

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Starting evaluation...")

    HF_TOKEN = os.environ.get("HF_TOKEN")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Load eval data
    vol.reload()
    dataset = load_from_disk(data_dir)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    # Load tokenizer
    token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **token_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_sql(model, prompt, max_new_tokens=128):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def normalize_sql(sql):
        """Extract the first SQL statement and normalize for comparison."""
        # Truncate at first ### or newline to isolate the SQL
        for stop in ["###", "\n"]:
            if stop in sql:
                sql = sql[:sql.index(stop)]
        return " ".join(sql.lower().split()).strip().rstrip(";")

    def build_prompt(example):
        return (
            "### Task: Generate a SQL query to answer the question.\n"
            f"### Schema:\n{example['context']}\n"
            f"### Question:\n{example['question']}\n"
            "### SQL:\n"
        )

    # -- Run base model --
    log.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, **token_kwargs, dtype=dtype, device_map="auto",
    )

    base_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(base_model, prompt)
        base_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Base model: {i + 1}/{len(eval_ds)}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Run fine-tuned model --
    log.info("Loading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_dir, dtype=dtype, device_map="auto",
    )

    ft_results = []
    for i, example in enumerate(eval_ds):
        prompt = build_prompt(example)
        generated = generate_sql(ft_model, prompt)
        ft_results.append(generated)
        if (i + 1) % 10 == 0:
            log.info(f"Fine-tuned model: {i + 1}/{len(eval_ds)}")

    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    base_correct = 0
    ft_correct = 0
    comparisons = []

    for i, example in enumerate(eval_ds):
        expected = example["answer"]
        base_gen = base_results[i]
        ft_gen = ft_results[i]

        base_match = normalize_sql(base_gen) == normalize_sql(expected)
        ft_match = normalize_sql(ft_gen) == normalize_sql(expected)

        if base_match:
            base_correct += 1
        if ft_match:
            ft_correct += 1

        comparisons.append({
            "question": example["question"],
            "schema": example["context"],
            "expected": expected,
            "base": base_gen,
            "finetuned": ft_gen,
            "base_correct": base_match,
            "ft_correct": ft_match,
        })

    total = len(eval_ds)
    base_acc = base_correct / total * 100
    ft_acc = ft_correct / total * 100
    improvement = ft_acc - base_acc

    log.info(f"Base model accuracy: {base_acc:.1f}% ({base_correct}/{total})")
    log.info(f"Fine-tuned accuracy: {ft_acc:.1f}% ({ft_correct}/{total})")

    # -- Build final eval report --
    improvement_badge = (
        f'<span class="badge badge-success">+{improvement:.1f}pp</span>'
        if improvement > 0
        else f'<span class="badge badge-danger">{improvement:.1f}pp</span>'
    )

    bar_chart = make_bar_chart(
        labels=["Exact Match Accuracy"],
        series={
            "Base Model": [base_acc],
            "Fine-Tuned": [ft_acc],
        },
        title="Base vs Fine-Tuned Accuracy",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=100.0,
    )

    examples_html = ""
    for c in comparisons[:10]:
        base_badge = '<span class="badge badge-success">correct</span>' if c["base_correct"] else '<span class="badge badge-danger">wrong</span>'
        ft_badge = '<span class="badge badge-success">correct</span>' if c["ft_correct"] else '<span class="badge badge-danger">wrong</span>'
        examples_html += f"""
        <div class="card">
          <p><b>Q:</b> {c['question']}</p>
          <p style="font-size:0.85em; color:#6c757d;"><b>Schema:</b> {c['schema'][:200]}...</p>
          <table>
            <tr><th>Source</th><th>SQL</th><th>Result</th></tr>
            <tr><td>Expected</td><td><code>{c['expected']}</code></td><td></td></tr>
            <tr><td>Base</td><td><code>{c['base'][:200]}</code></td><td>{base_badge}</td></tr>
            <tr><td>Fine-tuned</td><td><code>{c['finetuned'][:200]}</code></td><td>{ft_badge}</td></tr>
          </table>
        </div>"""

    report_html = wrap_report(
        f"<h2>Evaluation Results</h2>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{base_acc:.1f}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{ft_acc:.1f}%</div><div class="label">Fine-Tuned Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{improvement:+.1f}pp</div><div class="label">Improvement</div></div>'
        f'  <div class="stat"><div class="value">{total}</div><div class="label">Eval Examples</div></div>'
        f'</div>'
        f'<div class="chart-container">{bar_chart}</div>'
        f'<h3>Example Comparisons {improvement_badge}</h3>'
        f'{examples_html}'
        f'<div class="note">'
        f'<b>Note:</b> Exact match accuracy compares normalized SQL output. '
        f'The fine-tuned model may generate semantically correct queries that differ in formatting.'
        f'</div>'
    )

    metrics = {
        "base_accuracy": round(base_acc, 1),
        "finetuned_accuracy": round(ft_acc, 1),
        "improvement": round(ft_acc - base_acc, 1),
        "num_examples": total,
        "comparisons": comparisons[:10],
    }

    return {"metrics": metrics, "report_html": report_html}


# ------------------------------------------------------------------
# Pipeline: orchestrate everything
# ------------------------------------------------------------------


@app.function(cpu=2, memory=4096, volumes={DATA_PATH: vol}, timeout=14400)
def pipeline(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "b-mc2/sql-create-context",
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict:
    """
    End-to-end LLM fine-tuning pipeline.

    1. Download and format dataset
    2. Fine-tune model (full / LoRA / QLoRA)
    3. Evaluate: before/after comparison on test set

    Returns the fine-tuned model directory (on the shared volume) so it can be
    served directly, plus the training / evaluation / pipeline HTML reports.
    """
    log.info(f"Pipeline: {model_name} | method={method} | dataset={dataset_name}")
    steps = ["Prepare Data", "Train", "Evaluate"]

    method_badge = f'<span class="badge badge-info">{method.upper()}</span>'

    # Step 1: Prepare data
    data_dir = prepare_data.remote(dataset_name, max_train_samples, max_eval_samples)

    # Step 2: Train
    train_result = train.remote(
        model_name, data_dir, method, epochs, lr, batch_size, lora_r, lora_alpha,
    )
    finetuned_dir = train_result["model_dir"]

    # Step 3: Evaluate
    eval_result = evaluate.remote(model_name, finetuned_dir, data_dir, num_eval_examples)
    metrics = eval_result["metrics"]

    # Final pipeline report
    improvement = metrics["improvement"]
    improvement_badge = (
        f'<span class="badge badge-success">+{improvement:.1f}pp</span>'
        if improvement > 0
        else f'<span class="badge badge-danger">{improvement:.1f}pp</span>'
    )

    pipeline_report = wrap_report(
        f"<h2>Pipeline Complete</h2>"
        f"<h3>{model_name} {method_badge}</h3>"
        f'{pipeline_step_indicator(3, steps)}'
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{metrics["base_accuracy"]}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{metrics["finetuned_accuracy"]}%</div><div class="label">Fine-Tuned Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{improvement:+.1f}pp</div><div class="label">Improvement {improvement_badge}</div></div>'
        f'  <div class="stat"><div class="value">{method.upper()}</div><div class="label">Method</div></div>'
        f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
        f'  <div class="stat"><div class="value">{metrics["num_examples"]}</div><div class="label">Eval Examples</div></div>'
        f'</div>'
        f'<div class="note">'
        f'See <b>training_report.html</b> for training loss/LR charts, '
        f'and <b>evaluation_report.html</b> for detailed example comparisons.'
        f'</div>'
    )

    log.info(f"Pipeline complete. Improvement: {metrics['improvement']:+.1f}pp")
    return {
        "model_dir": finetuned_dir,
        "metrics": metrics,
        "pipeline_report": pipeline_report,
        "training_report": train_result["report_html"],
        "evaluation_report": eval_result["report_html"],
    }


# ------------------------------------------------------------------
# Local entrypoint — drives the pipeline and writes reports to disk
# ------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "b-mc2/sql-create-context",
    method: str = "lora",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    max_train_samples: int = 5000,
    max_eval_samples: int = 500,
    num_eval_examples: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Run the full fine-tuning pipeline on Modal and save the reports locally."""
    result = pipeline.remote(
        model_name=model_name,
        dataset_name=dataset_name,
        method=method,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        num_eval_examples=num_eval_examples,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    for fname, key in [
        ("training_report.html", "training_report"),
        ("evaluation_report.html", "evaluation_report"),
        ("report.html", "pipeline_report"),
    ]:
        with open(fname, "w") as f:
            f.write(result[key])
        print(f"Wrote {fname}")

    metrics = result["metrics"]
    print(json.dumps({k: v for k, v in metrics.items() if k != "comparisons"}, indent=2))
    print(f"Fine-tuned model saved to volume 'lora-qlora' at {result['model_dir']}")
    print("Serve it with:  modal serve serve.py   (or  modal deploy serve.py)")
