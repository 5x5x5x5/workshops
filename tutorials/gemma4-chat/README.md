# Gemma 4 Chat on Modal

Four-app port of the original Ollama+Gradio Gemma 4 demos to [Modal](https://modal.com), running vLLM + Gradio on cloud GPUs:

- **vLLM model server** (`vllm_server.py`) — serves Gemma 4 IT via vLLM's OpenAI-compatible API, as a Modal web endpoint. Caches weights in a `modal.Volume`. Autoscales to zero.
- **Gradio chat UI** (`chat_app.py`) — text chat, served as a Modal ASGI app. Talks to the vLLM web endpoint. Has a thinking-mode toggle and a thinking-budget slider.
- **Gradio vision UI** (`vision_app.py`) — upload an image, ask questions, or get emergent bounding-box detections drawn on the image. Same vLLM backend.
- **Gradio live-camera UI** (`live_camera_app.py`) — webcam → vision caption every few seconds. Same vLLM backend (Gemma 4 is multimodal). Modal serves over HTTPS, so the webcam works from any browser.

All three Gradio apps preserve the 🧠 Thinking panel from the originals — Gemma 4 IT's thinking is wrapped in `<|channel>...<channel|>` special-token markers, which we keep visible in the response by setting `skip_special_tokens=False` and parse client-side.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Model + GPU choice, shared secret/Volume, vLLM URL resolver. Default is `gemma-4-26B-A4B-it`; flip via `GEMMA_VARIANT=31b`. |
| `prefetch_model.py` | One-shot download of HF weights into the `hf-cache` Modal Volume. |
| `vllm_server.py` | vLLM OpenAI server as a Modal `@modal.web_server`. |
| `chat_app.py` | Gradio chat UI as a Modal `@modal.asgi_app`. |
| `vision_app.py` | Gradio image Q&A + detection UI. |
| `live_camera_app.py` | Gradio webcam vision-caption UI. |
| `requirements.txt` | Just `modal` (everything else is declared inline in each image). |
| `SPARK_SETUP.md` | Modal account + secret + prefetch quick-start. |

## Setup

```bash
cd tutorials/gemma4-chat

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## Add your HF token

Gemma 4 is gated. Create a Modal secret with a HuggingFace token whose account has accepted the Gemma license:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

(Or create it in the Modal dashboard under **Secrets**.)

## Prefetch the weights (recommended)

Download the gated Gemma weights into the `hf-cache` Volume once, so the vLLM server doesn't pay the download on its first cold start:

```bash
modal run prefetch_model.py
# 31B variant:
GEMMA_VARIANT=31b modal run prefetch_model.py
```

## Deploy

Order matters — deploy the vLLM server first so the Gradio apps can resolve its URL by name.

```bash
# 1. Serve Gemma 4 via vLLM (persistent URL).
modal deploy vllm_server.py

# 2. Deploy the Gradio front-ends.
modal deploy chat_app.py
modal deploy vision_app.py
modal deploy live_camera_app.py
```

Each `modal deploy` prints the app's public URL. Open the chat/vision/camera URL in a browser. The first request spins up the vLLM replica (cold start — a few minutes on first load after idle), subsequent requests are warm.

### Iterating locally

Use `modal serve <file>.py` instead of `modal deploy` for a hot-reloading dev URL that tears down when you Ctrl-C:

```bash
modal serve vllm_server.py     # ephemeral vLLM URL
modal serve chat_app.py        # hot-reloads on save
```

When running a front-end against an ad-hoc `modal serve vllm_server.py`, point it at that URL explicitly:

```bash
VLLM_URL=https://<your-workspace>--gemma4-26b-a4b-it-vllm-serve.modal.run modal serve chat_app.py
```

Otherwise the front-ends resolve the **deployed** vLLM app's URL by name.

## Switching models

```bash
GEMMA_VARIANT=31b modal deploy vllm_server.py
GEMMA_VARIANT=31b modal deploy chat_app.py
```

| Variant | Params | GPU spec in `config.py` | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (default) | 26B total / 4B active (MoE) | `A100` | Fast — only 4B active params per forward pass. ~52 GB weights fit on one 80 GB A100. |
| `gemma-4-31B` | 31B dense | `A100:2` | Dense bf16 ≈ 62 GB; tensor-parallel across 2 A100s. `serve()` reads the `:2` count and passes `--tensor-parallel-size 2`. |

The `gpu` field in `config.py` uses Modal's `"<gpu>"` / `"<gpu>:<count>"` format. Edit it for different hardware (`"A10G"`, `"H100"`, `"A100-80GB:2"`, etc.).

## Architecture

```
┌────────────────────┐     resolve URL by name    ┌──────────────────────┐
│  gemma4-chat-ui    │  ─────────────────────────▶│ gemma4-26b-a4b-vllm  │
│  (Gradio ASGI,CPU) │   Function.from_name(...)   │ (vLLM web_server,GPU)│
│                    │   .get_web_url() → /v1      │ port 8000            │
└────────────────────┘                             └──────────────────────┘
        ▲                                                    ▲
        │ user (HTTPS)                                       │ load weights
        │                                                    │
   browser                                              hf-cache Volume
                                                        (prefetched HF weights)
```

The vision and live-camera apps talk to the same vLLM endpoint the same way.

## Why vLLM (not Ollama)?

vLLM exposes an OpenAI-compatible API, handles GPU batching, and streams cleanly. On Modal it runs as a `@modal.web_server` GPU function that autoscales to zero, so you only pay for GPU time while requests are in flight. Ollama would mean managing a sidecar process and manual model pulls inside the container with no scale-to-zero.

## Troubleshooting

**`Repository google/gemma-4-26B-A4B does not exist in HuggingFace`** — your HF token hasn't accepted the Gemma license, or the repo path drifted. Visit the model page and click "Acknowledge license", then re-create the `huggingface-secret`.

**Front-end can't reach vLLM / `get_web_url` errors** — the vLLM app must be **deployed** (`modal deploy vllm_server.py`) before the front-ends can resolve it by name. For an ad-hoc `modal serve` vLLM, set `VLLM_URL` on the front-end (see *Iterating locally*).

**vLLM OOMs at startup** — drop `--max-model-len` in `config.py`, lower `--gpu-memory-utilization` in `vllm_server.py`, or move to a larger GPU / add a `:2` count for tensor parallelism.

**`<think>` tags showing inline in the answer** — Gemma chose not to produce a thinking block for that prompt, or the tag name differs. Check what the model actually emits via vLLM's `/docs` UI, then update `OPEN`/`CLOSE` in `_split_thinking`.

**Chat UI shows the URL but the first request hangs** — the vLLM replica is cold-starting (image pull + weight load + kernel compile). Watch the vLLM app logs in the Modal dashboard.

**vLLM image / Gemma 4 architecture** — Gemma 4's architecture is too new for vanilla vLLM, so `vllm_server.py` starts from NVIDIA's prebuilt `vllm/vllm-openai:gemma4-cu130` image (via `modal.Image.from_registry`), which bundles the matching vLLM patches. See [build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

## Notes on the port

- **Gradio version matters** — `gr.Chatbot(type="messages")` needs Gradio 5.x; 6.x dropped that kwarg. We pin `gradio==5.42.0`.
- **Webcam needs HTTPS** — `getUserMedia` is blocked over plain HTTP from a non-localhost origin. Modal serves every web app over HTTPS, so the live-camera webcam works from any browser with no tunnel (the original needed a `gradio.live` share tunnel for this).
- **Scale-to-zero** — the vLLM function scales down after 30 min idle (`scaledown_window=1800`); the front-ends after 15 min. First request after idle pays the cold start.
- **Font for detection labels** — `vision_app.py` installs `fonts-dejavu-core` via `apt_install` so `DejaVuSans-Bold.ttf` is available for drawing box labels.
