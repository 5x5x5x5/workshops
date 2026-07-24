"""
Serve the fine-tuned model as a FastAPI endpoint on Modal.

The trained model lives on the shared `lora-qlora` volume (written by
workflow.py). This app loads it once per container and exposes a `/generate`
endpoint for text-to-SQL.

Usage:
    # Dev server with a live-reloading URL
    modal serve serve.py

    # Deploy a persistent endpoint
    modal deploy serve.py

    # Test the endpoint (URL is printed when you serve/deploy)
    curl -X POST https://<your-workspace>--finetuned-sql-api-model-generate.modal.run \
      -H "Content-Type: application/json" \
      -d '{"schema": "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)",
           "question": "What is the average salary by department?"}'
"""

import modal
from pydantic import BaseModel, Field

from config import DATA_PATH, vol

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
        "fastapi[standard]",
        "pydantic",
    )
)

app = modal.App("finetuned-sql-api", image=image)


class SQLRequest(BaseModel):
    model_config = {"populate_by_name": True}

    schema_: str | None = Field(None, alias="schema")
    question: str

    @property
    def context(self) -> str:
        return self.schema_ or ""


class SQLResponse(BaseModel):
    sql: str
    raw_output: str


@app.cls(
    gpu="A10G",
    volumes={DATA_PATH: vol},
    scaledown_window=1800,  # keep a warm container for 30 minutes
    min_containers=0,
)
class Model:
    @modal.enter()
    def load(self):
        """Load the fine-tuned model once when the container starts."""
        import logging
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        vol.reload()
        model_path = Path(MODEL_DIR)

        if model_path.exists():
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path), dtype=dtype, device_map="auto",
            )
            self.model.eval()
            self.logger.info("Model loaded successfully")
        else:
            self.logger.warning(f"Model not found at {model_path}")
            self.model = None
            self.tokenizer = None

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "healthy" if self.model is not None else "not_ready",
            "model_loaded": self.model is not None,
        }

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: SQLRequest) -> SQLResponse:
        import torch
        from fastapi import HTTPException

        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        prompt = (
            "### Task: Generate a SQL query to answer the question.\n"
            f"### Schema:\n{request.context}\n"
            f"### Question:\n{request.question}\n"
            "### SQL:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        # Extract just the SQL (first line before ### or newline)
        sql = raw
        for stop in ["###", "\n"]:
            if stop in sql:
                sql = sql[:sql.index(stop)]
        sql = sql.strip()

        return SQLResponse(sql=sql, raw_output=raw)
