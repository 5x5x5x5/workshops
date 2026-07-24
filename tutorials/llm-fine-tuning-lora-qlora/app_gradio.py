"""
Text-to-SQL — Gradio frontend on Modal.

Type a question and a schema, get a SQL query from the fine-tuned model.
The model is loaded once per container directly from the shared `lora-qlora`
volume (written by workflow.py) — no separate server required.

Usage:
    # Dev server with a live-reloading URL
    modal serve app_gradio.py

    # Deploy a persistent URL
    modal deploy app_gradio.py
"""

import modal

from config import DATA_PATH, hf_secret, vol

MODEL_DIR = f"{DATA_PATH}/finetuned_model"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "gradio>=5.0.0",
        "fastapi[standard]",
    )
)

app = modal.App("sql-generator-ui", image=image)


# ------------------------------------------------------------------
# Example schemas & questions
# ------------------------------------------------------------------

EXAMPLE_SCHEMAS = [
    "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
    "CREATE TABLE orders (order_id INT, customer_id INT, product VARCHAR, quantity INT, price DECIMAL, order_date DATE)",
    "CREATE TABLE students (student_id INT, name VARCHAR, grade VARCHAR, gpa FLOAT, major VARCHAR)",
]

EXAMPLE_QUESTIONS = [
    ("What is the average salary by department?", EXAMPLE_SCHEMAS[0]),
    ("Which product had the most orders?", EXAMPLE_SCHEMAS[1]),
    ("List students with a GPA above 3.5", EXAMPLE_SCHEMAS[2]),
    ("What is the total revenue per product?", EXAMPLE_SCHEMAS[1]),
    ("How many employees are in each department?", EXAMPLE_SCHEMAS[0]),
]


# ------------------------------------------------------------------
# Gradio app served as an ASGI web endpoint
# ------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={DATA_PATH: vol},
    secrets=[hf_secret],
    scaledown_window=1800,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import gradio as gr
    import torch
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the fine-tuned model once per container.
    vol.reload()
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map="auto")
    model.eval()

    def generate_sql(schema: str, question: str):
        if not question.strip():
            return "", "", "Please enter a question."

        prompt = (
            "### Task: Generate a SQL query to answer the question.\n"
            f"### Schema:\n{schema}\n"
            f"### Question:\n{question}\n"
            "### SQL:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        sql = raw
        for stop in ["###", "\n"]:
            if stop in sql:
                sql = sql[:sql.index(stop)]
        sql = sql.strip()

        return sql, raw, ""

    with gr.Blocks(title="Text-to-SQL Generator") as demo:
        gr.Markdown(
            "# Text-to-SQL Generator\n"
            "Enter a database schema and a natural language question to generate a SQL query.\n\n"
            "Powered by a fine-tuned SmolLM2-135M model trained with LoRA on the "
            "[sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) dataset."
        )

        with gr.Row():
            with gr.Column(scale=1):
                schema_input = gr.Textbox(
                    label="Database Schema",
                    placeholder="CREATE TABLE employees (id INT, name VARCHAR, ...)",
                    lines=4,
                    value=EXAMPLE_SCHEMAS[0],
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the average salary by department?",
                    lines=2,
                )
                generate_btn = gr.Button("Generate SQL", variant="primary", size="lg")

            with gr.Column(scale=1):
                sql_output = gr.Code(
                    label="Generated SQL",
                    language="sql",
                    lines=4,
                )
                raw_output = gr.Textbox(
                    label="Raw Model Output",
                    lines=6,
                    interactive=False,
                )
                error_output = gr.Textbox(
                    label="",
                    visible=False,
                )

        gr.Examples(
            examples=[[s, q] for q, s in EXAMPLE_QUESTIONS],
            inputs=[schema_input, question_input],
            label="Examples — click to load",
        )

        generate_btn.click(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=[sql_output, raw_output, error_output],
        )

        question_input.submit(
            generate_sql,
            inputs=[schema_input, question_input],
            outputs=[sql_output, raw_output, error_output],
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
