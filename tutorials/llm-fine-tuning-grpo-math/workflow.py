"""
GRPO Fine-Tuning — Teach a model to solve grade-school math (GSM8K).

GRPO (Group Relative Policy Optimization) generates multiple completions per
prompt, scores them with a reward function, and reinforces the best ones.

Here the reward is simple: did the model reach the correct final answer?
Multiple valid chains of reasoning exist for each problem, so GRPO can explore
different solution paths — unlike SFT which teaches one fixed derivation.

This is the sibling of the `llm-fine-tuning-grpo-code` tutorial. The big
differences:

  - Reward is *answer correctness* (parse the final number, compare to gold),
    so there is NO sandbox — nothing untrusted is executed.
  - The dataset is GSM8K, which a small instruct model already solves ~30-40%
    of the time. That matters: GRPO can only sharpen capability the base model
    already has, so a dataset in the model's "learnable zone" is essential.
    (Competition sets like DeepMath are ~0% for a 0.5B model — all-impossible,
    no signal. Same lesson the code tutorial learned the hard way.)

Usage:
    # Quick sanity check
    modal run workflow.py --model-name "Qwen/Qwen2.5-0.5B-Instruct" \\
        --max-train-samples 20 --epochs 1 --num-generations 2 --batch-size 2 \\
        --max-completion-length 128 --num-eval-examples 8

    # Workshop run (single T4, ~20-30 min)
    modal run workflow.py

    # Full fine-tuning (no LoRA)
    modal run workflow.py --method full
"""

import json
import logging
import os
import re
import shutil

from config import app, hf_secret, vol, DATA_PATH, GPU, GPU_MEMORY, CPU_MEMORY
from report_helpers import make_bar_chart, make_line_chart, pipeline_step_indicator, wrap_report

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


DEFAULT_DATASET = "openai/gsm8k"

# The system prompt asks for step-by-step reasoning ending in \boxed{...}, which
# gives the reward function a reliable place to find the final answer.
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give the final answer on its own line in the form \\boxed{answer}."
)


# ------------------------------------------------------------------
# Answer parsing + reward (pure functions — no sandbox needed)
# ------------------------------------------------------------------

def _completion_text(completion) -> str:
    """GRPO completions are conversational (a list of message dicts) when the
    prompt is conversational; pull the assistant text out. Fall back to str."""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return completion or ""


def _to_number(s: str):
    """Best-effort parse of a numeric answer string → float, or None."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").replace("\\", "")
    m = re.search(r"-?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def extract_pred(text: str):
    """Extract the model's predicted number: prefer \\boxed{...}, then '#### x',
    then the last number in the text."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return _to_number(boxed[-1])
    hashed = re.findall(r"####\s*([^\n]+)", text)
    if hashed:
        return _to_number(hashed[-1])
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text)
    return _to_number(nums[-1]) if nums else None


def is_correct(completion_text: str, gold: str) -> bool:
    pred = extract_pred(completion_text)
    gold_num = _to_number(gold)
    return pred is not None and gold_num is not None and abs(pred - gold_num) < 1e-4


# ------------------------------------------------------------------
# Task 1: Prepare dataset
# ------------------------------------------------------------------

@app.function(cpu=2, memory=CPU_MEMORY, timeout=1800, volumes={DATA_PATH: vol})
async def prepare_data(
    dataset_name: str = DEFAULT_DATASET,
    max_train_samples: int = 200,
    max_eval_samples: int = 200,
) -> str:
    """Load GSM8K and build (question, gold-answer) train/eval splits.

    GSM8K columns: question, answer. The gold answer is the number after the
    '####' marker in the reference solution. We keep the raw question and the
    gold number; the conversational prompt is assembled at train time.

    Writes the processed splits to the shared volume and returns the path.
    """
    from datasets import Dataset, DatasetDict, load_dataset

    log.info(f"Loading dataset: {dataset_name}")

    def parse_split(split_rows):
        rows = []
        for row in split_rows:
            # GSM8K: question + answer ("...\n#### 42"). Other datasets: try
            # common column names and treat the answer field as the gold value.
            if "question" in row:
                question = row["question"]
            elif "problem" in row:
                question = row["problem"]
            else:
                continue

            answer_field = row.get("answer") or row.get("solution") or row.get("final_answer") or ""
            hashed = re.findall(r"####\s*([^\n]+)", str(answer_field))
            gold = hashed[-1].strip() if hashed else str(answer_field).strip()
            gold_num = _to_number(gold)
            if gold_num is None:
                continue

            rows.append({"question": question, "gold": str(gold)})
        return rows

    if "gsm8k" in dataset_name:
        ds = load_dataset(dataset_name, "main")
        train_rows = parse_split(ds["train"])
        eval_rows = parse_split(ds["test"])
    else:
        ds = load_dataset(dataset_name, split="train")
        all_rows = parse_split(ds)
        import random
        random.Random(42).shuffle(all_rows)
        split = min(max_eval_samples, len(all_rows) // 5)
        eval_rows, train_rows = all_rows[:split], all_rows[split:]

    import random
    random.Random(42).shuffle(train_rows)
    random.Random(7).shuffle(eval_rows)

    n_train = min(max_train_samples, len(train_rows))
    n_eval = min(max_eval_samples, len(eval_rows))

    processed = DatasetDict({
        "train": Dataset.from_list(train_rows[:n_train]),
        "eval": Dataset.from_list(eval_rows[:n_eval]),
    })

    output_dir = os.path.join(DATA_PATH, "dataset")
    shutil.rmtree(output_dir, ignore_errors=True)
    processed.save_to_disk(output_dir)
    vol.commit()
    log.info(f"Dataset ready: {n_train} train, {n_eval} eval")

    return output_dir


# ------------------------------------------------------------------
# Task 2: Train with GRPO
# ------------------------------------------------------------------

@app.function(gpu=GPU, memory=GPU_MEMORY, timeout=3600, volumes={DATA_PATH: vol}, secrets=[hf_secret])
async def train(
    model_name: str,
    data_path: str,
    method: str = "lora",
    epochs: int = 1,
    lr: float = 1e-5,
    batch_size: int = 6,
    num_generations: int = 6,
    max_completion_length: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.005,
) -> tuple[str, str]:
    """Fine-tune a model with GRPO — reward = did it reach the right answer?

    Saves the trained model to the shared volume and returns
    ``(model_dir, training_report_html)``.

    Args:
        beta: KL penalty coefficient anchoring the policy to the base model.
            Prevents drift/overfit (train reward up, held-out down). ~0.001-0.005
            is standard for small-model GSM8K GRPO; 0 disables the KL term.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    HF_TOKEN = os.environ.get("HF_TOKEN")

    log.info(f"GRPO Math Training: model={model_name}, method={method}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # -- Load data --
    dataset = load_from_disk(data_path)

    # -- Load model + tokenizer --
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # T4 (Turing) has no bf16 — fall back to fp16, not fp32.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    train_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    )
    log.info(f"Training precision: {train_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        dtype=train_dtype,
        device_map="auto",
    )

    # -- Build conversational prompts (system + question). GRPOTrainer applies
    #    the chat template; extra columns like `gold` are forwarded to rewards. --
    def to_conversational(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ]
        }

    train_ds = dataset["train"].map(to_conversational)

    # -- LoRA (optional) --
    peft_config = None
    if method == "lora":
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # -- Metrics tracking --
    training_log: list[dict] = []
    reward_log: list[dict] = []
    reward_stats = {"calls": 0, "correct": 0, "total": 0}

    # -- Reward functions (sync + pure). They record metrics as training runs. --
    def accuracy_reward(completions, gold, **kwargs) -> list[float]:
        rewards = []
        batch_correct = 0
        for completion, g in zip(completions, gold):
            correct = is_correct(_completion_text(completion), g)
            rewards.append(1.0 if correct else 0.0)
            if correct:
                batch_correct += 1
                reward_stats["correct"] += 1
            reward_stats["total"] += 1

        reward_stats["calls"] += 1
        running_acc = reward_stats["correct"] / max(reward_stats["total"], 1) * 100
        batch_acc = batch_correct / len(completions) * 100
        reward_log.append({
            "batch": reward_stats["calls"],
            "accuracy": round(running_acc, 2),
            "batch_accuracy": round(batch_acc, 2),
        })
        if reward_stats["calls"] % 5 == 0:
            log.info(
                f"[Reward] batch {reward_stats['calls']}: "
                f"accuracy={running_acc:.1f}%, batch={batch_acc:.1f}%"
            )
        return rewards

    def format_reward(completions, **kwargs) -> list[float]:
        """Light bonus for producing a parseable \\boxed{...} answer."""
        return [
            1.0 if re.search(r"\\boxed\{[^}]*\}", _completion_text(c)) else 0.0
            for c in completions
        ]

    # -- Trainer callback for loss/progress logging --
    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
            }
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 4)
            if "grad_norm" in logs:
                entry["grad_norm"] = round(float(logs["grad_norm"]), 4)
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f}"
                + (f" loss={entry['loss']:.4f}" if "loss" in entry else "")
            )

    # -- GRPO config --
    output_dir = os.path.join(DATA_PATH, "checkpoints")
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,  # KL anchor to the base model — prevents overfit-drift
        reward_weights=[1.0, 0.2],  # accuracy dominates; format is a nudge
        logging_steps=5,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=[accuracy_reward, format_reward],
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[MetricsCallback()],
    )

    log.info(f"Starting GRPO training (num_generations={num_generations})...")
    trainer.train()
    log.info("Training loop finished.")

    final_acc = reward_stats["correct"] / max(reward_stats["total"], 1) * 100
    log.info(f"GRPO training complete. Train-time accuracy: {final_acc:.1f}%")

    # -- Save model --
    save_dir = os.path.join(DATA_PATH, "grpo_model")
    shutil.rmtree(save_dir, ignore_errors=True)
    if method == "lora":
        log.info("Merging LoRA weights...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_dir)
    else:
        log.info("Saving full model...")
        trainer.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    vol.commit()
    log.info("Model saved.")

    # -- Build the training report (reward/loss charts) --
    final_charts = ""
    loss_entries = [e for e in training_log if "loss" in e]
    if len(loss_entries) >= 2:
        loss_chart = make_line_chart(
            data=loss_entries, x_key="epoch", y_keys=["loss"],
            title="Training Loss", x_label="Epoch", y_label="Loss", colors=["#5a7db5"],
        )
        final_charts += f'<div class="chart-container">{loss_chart}</div>'
    if len(reward_log) >= 2:
        acc_chart = make_line_chart(
            data=reward_log, x_key="batch", y_keys=["accuracy", "batch_accuracy"],
            title="Answer Accuracy (Reward)", x_label="Reward Batch", y_label="Accuracy %",
            colors=["#06d6a0", "#5a7db5"], y_max_cap=105.0,
            y_display_names={"accuracy": "Running Avg", "batch_accuracy": "Per Batch"},
        )
        final_charts += f'<div class="chart-container">{acc_chart}</div>'

    report_html = wrap_report(
        f"<h2>GRPO Math Training Complete</h2>"
        f"<h3>{model_name}</h3>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{final_acc:.1f}%</div><div class="label">Train Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>'
        f'  <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>'
        f'</div>'
        f"{final_charts}"
    )

    return save_dir, report_html


# ------------------------------------------------------------------
# Task 3: Evaluate
# ------------------------------------------------------------------

@app.function(gpu=GPU, memory=GPU_MEMORY, timeout=3600, volumes={DATA_PATH: vol}, secrets=[hf_secret])
async def evaluate(
    model_name: str,
    finetuned_dir: str,
    data_path: str,
    num_examples: int = 40,
    max_new_tokens: int = 256,
) -> tuple[str, str]:
    """Compare base vs GRPO-trained model on held-out GSM8K problems.

    Returns ``(metrics_json, eval_report_html)``.
    """
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_TOKEN = os.environ.get("HF_TOKEN")

    log.info("Starting evaluation...")

    dataset = load_from_disk(data_path)
    eval_ds = dataset["eval"].select(range(min(num_examples, len(dataset["eval"]))))

    questions = eval_ds["question"]
    golds = eval_ds["gold"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    def generate(model, questions):
        results = []
        for q in questions:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append(gen)
        return results

    # -- Base model --
    log.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, dtype=dtype, device_map="auto")
    base_results = generate(base_model, questions)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- GRPO model --
    log.info("Loading GRPO-trained model...")
    ft_model = AutoModelForCausalLM.from_pretrained(finetuned_dir, dtype=dtype, device_map="auto")
    ft_results = generate(ft_model, questions)
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Score --
    base_correct = 0
    ft_correct = 0
    comparisons = []
    for i in range(len(questions)):
        b_ok = is_correct(base_results[i], golds[i])
        f_ok = is_correct(ft_results[i], golds[i])
        base_correct += int(b_ok)
        ft_correct += int(f_ok)
        comparisons.append({
            "question": questions[i][:200],
            "gold": golds[i],
            "base_pred": extract_pred(base_results[i]),
            "grpo_pred": extract_pred(ft_results[i]),
            "base_out": base_results[i][:300],
            "grpo_out": ft_results[i][:300],
            "base_ok": b_ok,
            "grpo_ok": f_ok,
        })

    total = len(questions)
    base_rate = base_correct / total * 100
    ft_rate = ft_correct / total * 100
    log.info(f"Base model: {base_rate:.1f}% ({base_correct}/{total})")
    log.info(f"GRPO model: {ft_rate:.1f}% ({ft_correct}/{total})")

    improvement = ft_rate - base_rate
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    bar_chart = make_bar_chart(
        labels=["Answer Accuracy"],
        series={"Base": [base_rate], "GRPO": [ft_rate]},
        title="Base vs GRPO — GSM8K",
        colors=["#adb5bd", "#0f3460"],
        y_max_cap=105.0,
    )

    examples_html = ""
    for c in comparisons[:15]:
        base_badge = "badge-success" if c["base_ok"] else "badge-danger"
        ft_badge = "badge-success" if c["grpo_ok"] else "badge-danger"
        examples_html += f"""
<div class="card">
<p><b>Q:</b> {c['question']}</p>
<p><b>Gold:</b> {c['gold']} |
  Base: <span class="badge {base_badge}">{c['base_pred']}</span> |
  GRPO: <span class="badge {ft_badge}">{c['grpo_pred']}</span></p>
</div>"""

    eval_report_html = wrap_report(
        f"<h2>Evaluation Results — GSM8K Math</h2>"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{base_rate:.1f}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{ft_rate:.1f}%</div><div class="label">GRPO Accuracy</div></div>'
        f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
        f'  <div class="stat"><div class="value">{total}</div><div class="label">Problems Tested</div></div>'
        f'</div>'
        f'<div class="chart-container">{bar_chart}</div>'
        f"<table>"
        f"<tr><th></th><th>Answer Accuracy</th></tr>"
        f"<tr><td><b>Base model</b></td><td>{base_rate:.1f}% ({base_correct}/{total})</td></tr>"
        f"<tr><td><b>GRPO-trained</b></td><td>{ft_rate:.1f}% ({ft_correct}/{total})</td></tr>"
        f"</table>"
        f"<h3>Examples</h3>"
        f"{examples_html}"
    )

    metrics_json = json.dumps({
        "base_accuracy": round(base_rate, 1),
        "grpo_accuracy": round(ft_rate, 1),
        "improvement": round(ft_rate - base_rate, 1),
        "num_problems": total,
        "comparisons": comparisons[:15],
    })

    return metrics_json, eval_report_html


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@app.function(cpu=2, memory=CPU_MEMORY, timeout=600)
def pipeline(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    method: str = "lora",
    epochs: int = 1,
    lr: float = 1e-5,
    batch_size: int = 6,
    num_generations: int = 6,
    max_completion_length: int = 256,
    dataset_name: str = DEFAULT_DATASET,
    max_train_samples: int = 200,
    max_eval_samples: int = 200,
    num_eval_examples: int = 40,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.005,
) -> dict:
    """
    GRPO fine-tuning pipeline — teach a model to solve GSM8K math.

    1. Prepare GSM8K (question + gold answer)
    2. Train with GRPO — reward = correct final answer
    3. Evaluate: accuracy before/after

    Returns a dict with the combined HTML report and the eval metrics JSON.
    """
    log.info(f"Pipeline: {model_name} | GRPO math ({dataset_name})")
    steps = ["Prepare Data", "GRPO Train", "Evaluate"]

    data_path = prepare_data.remote(dataset_name, max_train_samples, max_eval_samples)

    model_dir, training_html = train.remote(
        model_name, data_path, method, epochs, lr, batch_size,
        num_generations, max_completion_length, lora_r, lora_alpha, beta,
    )

    metrics_json, eval_html = evaluate.remote(
        model_name, model_dir, data_path, num_eval_examples, max_completion_length,
    )
    metrics = json.loads(metrics_json)

    improvement = metrics["improvement"]
    imp_badge = "badge-success" if improvement > 0 else "badge-danger" if improvement < 0 else "badge-info"

    summary_html = wrap_report(
        f"<h2>GRPO Math Pipeline Complete</h2>"
        f"<h3>{model_name}</h3>"
        f"{pipeline_step_indicator(3, steps)}"
        f'<div class="stat-grid">'
        f'  <div class="stat"><div class="value">{metrics["base_accuracy"]}%</div><div class="label">Base Accuracy</div></div>'
        f'  <div class="stat"><div class="value">{metrics["grpo_accuracy"]}%</div><div class="label">GRPO Accuracy</div></div>'
        f'  <div class="stat"><div class="value"><span class="badge {imp_badge}">{improvement:+.1f}pp</span></div><div class="label">Improvement</div></div>'
        f'</div>'
    )

    log.info(f"Pipeline complete. Improvement: {improvement:+.1f}pp")

    # Concatenate the per-stage reports into one self-contained HTML artifact.
    combined_html = summary_html + training_html + eval_html
    return {"report_html": combined_html, "metrics": metrics_json}


@app.local_entrypoint()
def main(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    method: str = "lora",
    epochs: int = 1,
    lr: float = 1e-5,
    batch_size: int = 6,
    num_generations: int = 6,
    max_completion_length: int = 256,
    dataset_name: str = DEFAULT_DATASET,
    max_train_samples: int = 200,
    max_eval_samples: int = 200,
    num_eval_examples: int = 40,
    lora_r: int = 16,
    lora_alpha: int = 32,
    beta: float = 0.005,
):
    """Run the GRPO math pipeline and write the HTML report + metrics locally."""
    result = pipeline.remote(
        model_name=model_name,
        method=method,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        dataset_name=dataset_name,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        num_eval_examples=num_eval_examples,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        beta=beta,
    )

    report_path = "grpo_math_report.html"
    with open(report_path, "w") as f:
        f.write(result["report_html"])

    metrics_path = "grpo_math_metrics.json"
    with open(metrics_path, "w") as f:
        f.write(result["metrics"])

    metrics = json.loads(result["metrics"])
    print(
        f"Base: {metrics['base_accuracy']}%  |  "
        f"GRPO: {metrics['grpo_accuracy']}%  |  "
        f"Improvement: {metrics['improvement']:+.1f}pp"
    )
    print(f"Wrote report to {report_path}")
    print(f"Wrote metrics to {metrics_path}")


# uv run modal run workflow.py
