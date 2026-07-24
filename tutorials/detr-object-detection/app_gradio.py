"""
RT-DETR Object Detection — Gradio frontend on Modal.

Upload images or use your webcam to detect objects with the fine-tuned RT-DETR
model served by app_server.py. The UI talks to the detection server over HTTP.

Usage:
    # Local dev with hot-reload (ephemeral URL)
    uv run modal serve app_gradio.py

    # Deploy a persistent URL
    uv run modal deploy app_gradio.py

The frontend auto-discovers the deployed detection server ("rtdetr-detection-server").
Override with a SERVER_URL env var to point at a different endpoint.
"""

import logging
import os

import modal

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Image & app
# ------------------------------------------------------------------

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gradio>=5.0.0",
    "httpx",
    "pillow>=10.0.0",
    "numpy",
    "fastapi[standard]",
)

app = modal.App("rtdetr-detection-ui", image=image)

# The deployed server app/class — used to look up the detection endpoint URL.
SERVER_APP_NAME = "rtdetr-detection-server"


def _resolve_server_url() -> str:
    """Find the detection server URL: SERVER_URL env override, else Modal lookup."""
    override = os.environ.get("SERVER_URL", "")
    if override:
        return override.rstrip("/")
    server = modal.Cls.from_name(SERVER_APP_NAME, "Server")()
    fn = server.fastapi_app
    url = fn.get_web_url() if hasattr(fn, "get_web_url") else fn.web_url
    return url.rstrip("/")


# ------------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------------

@app.function(cpu=2, memory=4096)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import base64
    import io

    import gradio as gr
    import httpx
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from PIL import Image, ImageDraw, ImageFont

    api_url = _resolve_server_url()
    log.info(f"Connecting to detection server: {api_url}")

    # -- Colors for different classes --
    COLORS = [
        "#0f3460", "#06d6a0", "#e94560", "#ffc107",
        "#5a7db5", "#ff6b6b", "#4ecdc4", "#45b7d1",
    ]

    def get_color(label_id: int) -> str:
        return COLORS[label_id % len(COLORS)]

    # -- Drawing helper --
    def _get_font(img_width: int):
        """Get a readable font, trying several common paths."""
        font_size = max(16, img_width // 35)
        for font_name in [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "Arial.ttf",
            "arial.ttf",
        ]:
            try:
                return ImageFont.truetype(font_name, size=font_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default(size=font_size)

    def draw_detections(image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        font = _get_font(img.width)
        line_width = max(3, img.width // 250)

        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = get_color(det["label_id"])
            label = det["label"]
            score = det["score"]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label background + text above the box
            caption = f"{label} {score:.0%}"
            text_bbox = draw.textbbox((x1, y1), caption, font=font)
            text_h = text_bbox[3] - text_bbox[1]
            # Position label above the box, or inside if at the top edge
            label_y = y1 - text_h - 4 if y1 - text_h - 4 > 0 else y1
            text_bbox = draw.textbbox((x1, label_y), caption, font=font)
            pad = 3
            draw.rectangle(
                [text_bbox[0] - pad, text_bbox[1] - pad,
                 text_bbox[2] + pad, text_bbox[3] + pad],
                fill=color,
            )
            draw.text((x1, label_y), caption, fill="white", font=font)

        return img

    # -- Detection function --
    def detect_image(image_path: str, threshold: float):
        """Send image to the detection server and return annotated image."""
        if image_path is None:
            return None, "No image provided."

        try:
            with open(image_path, "rb") as f:
                files = {"file": ("image.jpg", f, "image/jpeg")}
                response = httpx.post(
                    f"{api_url}/detect",
                    files=files,
                    params={"threshold": threshold},
                    timeout=30.0,
                )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return None, "Could not connect to detection server. Is it running?"
        except Exception as e:
            return None, f"Error: {e}"

        detections = result["detections"]
        image = Image.open(image_path).convert("RGB")
        annotated = draw_detections(image, detections)

        # Build summary text
        if not detections:
            summary = "No objects detected."
        else:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            parts = [f"{count}x {label}" for label, count in counts.items()]
            summary = f"Found {len(detections)} object(s): {', '.join(parts)}"

            # Per-detection details
            summary += "\n\n"
            for i, d in enumerate(detections):
                box = d["box"]
                summary += (
                    f"  {i+1}. {d['label']} ({d['score']:.0%}) "
                    f"[{box['x1']:.0f}, {box['y1']:.0f}, "
                    f"{box['x2']:.0f}, {box['y2']:.0f}]\n"
                )

        return annotated, summary

    # -- Webcam detection function --
    def detect_webcam(frame, threshold: float):
        """Send a webcam frame to the detection server."""
        if frame is None:
            return None, "No frame captured."

        try:
            # Convert numpy frame to JPEG bytes
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            response = httpx.post(
                f"{api_url}/detect_base64",
                json={"image": img_b64},
                params={"threshold": threshold},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return None, "Could not connect to detection server."
        except Exception as e:
            return None, f"Error: {e}"

        detections = result["detections"]
        annotated = draw_detections(img, detections)

        if not detections:
            summary = "No objects detected."
        else:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            parts = [f"{count}x {label}" for label, count in counts.items()]
            summary = f"Found {len(detections)} object(s): {', '.join(parts)}"

        return annotated, summary

    # -- Build the Gradio UI --
    with gr.Blocks(title="RT-DETR Object Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# RT-DETR Object Detection\n"
            "Upload an image or use your webcam to detect objects "
            "with the fine-tuned RT-DETR model."
        )

        with gr.Tab("Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        height=400,
                    )
                    upload_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    upload_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    upload_output = gr.Image(label="Detections", height=400)
                    upload_details = gr.Textbox(
                        label="Details",
                        lines=6,
                        interactive=False,
                    )

            upload_btn.click(
                detect_image,
                inputs=[upload_input, upload_threshold],
                outputs=[upload_output, upload_details],
            )

        with gr.Tab("Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam",
                        height=400,
                    )
                    webcam_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    webcam_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    webcam_output = gr.Image(label="Detections", height=400)
                    webcam_details = gr.Textbox(
                        label="Details",
                        lines=4,
                        interactive=False,
                    )

            webcam_btn.click(
                detect_webcam,
                inputs=[webcam_input, webcam_threshold],
                outputs=[webcam_output, webcam_details],
            )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
