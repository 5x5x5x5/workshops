"""
Serve the fine-tuned emotion classifier as a FastAPI endpoint on Modal.

The server loads the ModernBERT model from the shared model volume (produced by
`workflow.py`) once per container and exposes a `/predict` endpoint that returns
emotion predictions with confidence scores and attention weights for
visualization.

Usage:
    # Dev server with hot-reload (ephemeral URL, tails logs)
    uv run modal serve serve.py

    # Persistent deployment
    uv run modal deploy serve.py

    # Test the endpoint (URL printed by serve/deploy)
    curl -X POST https://<your-app-url>/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "I am so happy today!"}'
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers",
        "accelerate",
        "fastapi[standard]",
        "pydantic",
    )
)

app = modal.App("emotion-classifier-api", image=image)

# The fine-tuned model is written to this volume by workflow.py.
model_volume = modal.Volume.from_name("bert-emotion-model", create_if_missing=True)
MODEL_PATH = "/models"
FINETUNED_DIR = f"{MODEL_PATH}/finetuned_model"
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@app.cls(
    gpu="A10G",
    memory=8192,
    scaledown_window=300,
    volumes={MODEL_PATH: model_volume},
)
@modal.concurrent(max_inputs=10)
class EmotionClassifier:
    @modal.enter()
    def load_model(self):
        """Load the fine-tuned model once when the container starts."""
        import logging
        from pathlib import Path

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        model_path = Path(FINETUNED_DIR)
        if not model_path.exists():
            self.logger.warning(f"Model not found at {model_path}")
            self.model = None
            self.tokenizer = None
            return

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        # Use eager attention to extract attention weights.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            output_attentions=True,
            attn_implementation="eager",
        )
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.logger.info(f"Model loaded on {device}")

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel

        import torch

        web_app = FastAPI(
            title="Emotion Classifier",
            description="Classify emotions in text using a fine-tuned ModernBERT model",
            version="1.0.0",
        )

        class PredictRequest(BaseModel):
            text: str

        class EmotionScore(BaseModel):
            label: str
            score: float

        class PredictResponse(BaseModel):
            predicted_emotion: str
            confidence: float
            scores: list[EmotionScore]
            tokens: list[str]
            attention_weights: list[float]

        @web_app.get("/health")
        async def health():
            return {
                "status": "healthy" if self.model is not None else "not_ready",
                "model_loaded": self.model is not None,
            }

        @web_app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            if self.model is None:
                raise HTTPException(status_code=503, detail="Model not loaded")

            model = self.model
            tokenizer = self.tokenizer

            inputs = tokenizer(
                request.text, return_tensors="pt", truncation=True, max_length=128,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Probabilities
            probs = torch.softmax(outputs.logits[0], dim=-1).cpu().tolist()
            pred_idx = int(torch.argmax(outputs.logits[0]).item())

            # Attention: CLS token attention from last layer, averaged across heads
            last_attention = outputs.attentions[-1][0]  # (num_heads, seq_len, seq_len)
            cls_attention = last_attention.mean(dim=0)[0].cpu().tolist()  # CLS row

            # Clean tokens for display
            token_ids = inputs["input_ids"][0]
            raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)

            clean_tokens = []
            clean_attention = []
            for i, tok in enumerate(raw_tokens):
                if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
                    continue
                if tok == tokenizer.pad_token:
                    continue
                # Strip tokenizer prefixes
                display_tok = tok.replace("##", "").replace("Ġ", "").replace("▁", "")
                if not display_tok.strip():
                    continue
                clean_tokens.append(display_tok)
                clean_attention.append(cls_attention[i])

            # Normalize attention to 0-1 for visualization
            if clean_attention:
                max_att = max(clean_attention)
                min_att = min(clean_attention)
                att_range = max_att - min_att or 1
                clean_attention = [(a - min_att) / att_range for a in clean_attention]

            scores = [
                EmotionScore(label=EMOTION_LABELS[i], score=round(probs[i], 4))
                for i in range(len(EMOTION_LABELS))
            ]

            return PredictResponse(
                predicted_emotion=EMOTION_LABELS[pred_idx],
                confidence=round(probs[pred_idx], 4),
                scores=scores,
                tokens=clean_tokens,
                attention_weights=clean_attention,
            )

        return web_app
