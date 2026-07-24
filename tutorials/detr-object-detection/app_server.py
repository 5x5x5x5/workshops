"""
RT-DETR Object Detection — FastAPI model server on Modal.

Serves a fine-tuned RT-DETR model for object detection inference. Accepts image
uploads and returns bounding box predictions. The model is loaded once per
container from the shared model volume written by the training pipeline
(workflow.py).

Usage:
    # Local dev with hot-reload (ephemeral URL, tears down on Ctrl-C)
    uv run modal serve app_server.py

    # Deploy a persistent URL
    uv run modal deploy app_server.py
"""

import base64
import io
import logging

import modal
from config import FINETUNED_SUBDIR, MODEL_PATH, model_volume

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Image & app
# ------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.49.0",
        "pillow>=10.0.0",
        "fastapi[standard]",
        "python-multipart",
    )
)

app = modal.App("rtdetr-detection-server", image=image)

# HuggingFace token secret — injected as HF_TOKEN (harmless for public weights).
hf_secret = modal.Secret.from_name("huggingface-secret")


# ------------------------------------------------------------------
# Server — model loaded once per container, FastAPI mounted as an ASGI app
# ------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    cpu=4,
    memory=16384,
    volumes={MODEL_PATH: model_volume},
    secrets=[hf_secret],
    scaledown_window=1800,
    min_containers=0,
)
@modal.concurrent(max_inputs=100)
class Server:
    @modal.enter()
    def load(self):
        """Load the fine-tuned model and processor into GPU memory."""
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_volume.reload()
        model_dir = f"{MODEL_PATH}/{FINETUNED_SUBDIR}"

        import os
        if not os.path.exists(model_dir):
            log.warning(
                f"No model found at {model_dir} — run the training pipeline "
                "(workflow.py) first. Endpoints will return 503 until then."
            )
            self.model = None
            self.processor = None
            self.device = None
            self.id2label = None
            return

        log.info(f"Loading model from: {model_dir}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir).to(device)
        self.model.eval()
        self.device = device
        self.id2label = self.model.config.id2label
        log.info(f"Model loaded on {device} — classes: {self.id2label}")

    def _detect(self, image, threshold: float) -> dict:
        """Run detection on a PIL image and return the JSON-serializable result."""
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_size = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"].cpu().tolist(),
            results["labels"].cpu().tolist(),
            results["boxes"].cpu().tolist(),
        ):
            detections.append({
                "label": self.id2label.get(label, str(label)),
                "label_id": label,
                "score": round(score, 4),
                "box": {
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1),
                },
            })

        return {
            "detections": detections,
            "image_size": {"width": image.width, "height": image.height},
            "threshold": threshold,
        }

    @modal.asgi_app()
    def fastapi_app(self):
        import fastapi
        from PIL import Image

        web_app = fastapi.FastAPI(title="RT-DETR Object Detection API")

        @web_app.get("/health")
        async def health():
            model_loaded = self.model is not None
            return {
                "status": "ready" if model_loaded else "starting",
                "model_loaded": model_loaded,
            }

        @web_app.get("/classes")
        async def get_classes():
            if self.id2label is None:
                raise fastapi.HTTPException(status_code=503, detail="Model not loaded")
            return {"classes": self.id2label}

        @web_app.post("/detect")
        async def detect(
            file: fastapi.UploadFile = fastapi.File(...),
            threshold: float = 0.5,
        ):
            """Run object detection on an uploaded image.

            Returns bounding boxes in xyxy format with labels and scores.
            """
            if self.model is None:
                raise fastapi.HTTPException(status_code=503, detail="Model not loaded")

            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return self._detect(image, threshold)

        @web_app.post("/detect_base64")
        async def detect_base64(
            request: fastapi.Request,
            threshold: float = 0.5,
        ):
            """Run object detection on a base64-encoded image.

            Expects JSON body: {"image": "<base64 string>"}
            Useful for webcam frames sent from the Gradio frontend.
            """
            if self.model is None:
                raise fastapi.HTTPException(status_code=503, detail="Model not loaded")

            body = await request.json()
            image_b64 = body.get("image", "")
            if not image_b64:
                raise fastapi.HTTPException(status_code=400, detail="No image provided")

            # Strip data URL prefix if present
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            contents = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return self._detect(image, threshold)

        return web_app
