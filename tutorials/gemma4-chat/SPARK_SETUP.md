# Setup for the Gemma 4 chat tutorial (Modal)

This is the minimal, ordered checklist to go from nothing to the four apps in
this directory running on [Modal](https://modal.com) and serving chat.

> **What changed from the original:** this tutorial used to target a local
> **NVIDIA DGX Spark** (Grace-Blackwell GB10, aarch64) running a self-managed
> GPU cluster. On Modal there's no host to image, no local GPU driver / runtime
> to install, no cluster to start, and no arch/registry wrangling — Modal
> provisions the GPUs and runs the containers. So all the host-provisioning,
> `nvidia-smi`, container-runtime, and cluster-bootstrap steps are gone. What
> remains is: a Modal account, the HuggingFace secret, an optional prefetch,
> and deploying the apps.

## 0. Prerequisites

You need locally:

```bash
# uv (https://docs.astral.sh/uv/)
uv --version
```

That's it — no Docker, no GPU, no NVIDIA runtime. The GPUs live in Modal's cloud.

## 1. HuggingFace: token + Gemma 4 license

Gemma 4 weights are gated. On HuggingFace:

1. Visit `https://huggingface.co/google/gemma-4-26B-A4B-it` and click **Acknowledge license**.
2. Generate a read-only token at `https://huggingface.co/settings/tokens`.

Keep that token handy — you'll put it into a Modal secret in step 4.

## 2. Clone + venv + Modal CLI

```bash
cd tutorials/gemma4-chat

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# `modal` is a console script provided by the `modal` package
modal --version
```

## 3. Authenticate with Modal (one-time)

```bash
uv run modal setup
```

This opens a browser to log in / sign up. No account? Create one at
[modal.com](https://modal.com).

## 4. Create the HF token secret

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

(The token from step 1. You can also create it in the Modal dashboard under
**Secrets**.) The vLLM server and the prefetch job read `HF_TOKEN` from this
secret.

## 5. Prefetch the model (recommended)

Download the gated weights into the `hf-cache` Modal Volume once, so the vLLM
server doesn't pay the download on its first cold start (~52 GB for the default
26B-A4B model):

```bash
modal run prefetch_model.py
# 31B variant:
GEMMA_VARIANT=31b modal run prefetch_model.py
```

You can skip this — the vLLM server will download into the same Volume on its
first startup — but prefetching keeps that one-time cost out of the serving
path.

## 6. Deploy the vLLM model server

```bash
modal deploy vllm_server.py
```

Things to know:

- **Model**: `google/gemma-4-26B-A4B-it`. ~52 GB safetensors. The IT
  (instruction-tuned) variant — needed for chat format and thinking-mode
  support.
- **Image**: `vllm/vllm-openai:gemma4-cu130` (via `modal.Image.from_registry`).
  NVIDIA's Gemma 4 fork — vanilla vLLM doesn't recognize Gemma 4's architecture.
- **GPU**: `A100` for the default MoE model. The dense `31b` variant uses
  `A100:2` (tensor-parallel across 2 GPUs).
- **Cold start**: a few minutes on first load — image pull (cached after first),
  weight load, kernel compile, CUDA-graph capture.
- **Autoscale**: scales to 0 after 30 min idle (`scaledown_window=1800`). First
  request after that pays the full cold start.

The command prints the server's public URL, e.g.
`https://<workspace>--gemma4-26b-a4b-it-vllm-serve.modal.run`. Its OpenAI base
URL is that `+ /v1`, and OpenAPI docs are at `+ /docs`.

## 7. Deploy the Gradio front-ends

```bash
modal deploy chat_app.py
modal deploy vision_app.py
modal deploy live_camera_app.py
```

These are small CPU-only images (Gradio + the OpenAI client). Each resolves the
deployed vLLM app's URL **by name** at startup (via
`modal.Function.from_name(...).get_web_url()`), so no wiring is needed as long
as the vLLM app is deployed first.

Each command prints the app's public HTTPS URL. Open it in a browser.

## 8. Open the chat

Paste the printed chat URL into a browser and type a message. The 🧠 Thinking
panel fills with the model's reasoning, then the answer appears below it.

For the live-camera app, the webcam works directly — Modal serves over HTTPS, so
`getUserMedia`'s secure-context requirement is satisfied with no extra tunnel.

## Iterating with `modal serve`

For a hot-reloading dev URL that tears down on Ctrl-C, use `modal serve` instead
of `modal deploy`:

```bash
modal serve vllm_server.py    # ephemeral vLLM URL
modal serve chat_app.py       # hot-reloads on save
```

To point a `modal serve` front-end at an ad-hoc `modal serve` vLLM URL, set
`VLLM_URL`:

```bash
VLLM_URL=https://<workspace>--gemma4-26b-a4b-it-vllm-serve.modal.run modal serve chat_app.py
```

## Values cheat sheet

| Setting | Value | Why |
|---|---|---|
| `gpu` in `config.py` | `A100` (MoE) / `A100:2` (dense 31B) | 52 GB / 62 GB of weights + KV cache |
| `--gpu-memory-utilization` | `0.90` | vLLM default; dedicated A100 has the headroom |
| vLLM base image | `vllm/vllm-openai:gemma4-cu130` | Gemma 4 architecture support |
| HF model | `google/gemma-4-26B-A4B-it` (IT variant) | IT ships `chat_template.jinja` for `/v1/chat/completions` |
| weight cache | `hf-cache` Modal Volume at `/root/.cache/huggingface` | download the gated weights once |
| inter-app URL | `Function.from_name(app, "serve").get_web_url()` | front-ends resolve the vLLM URL by name |

## Tear down

Modal apps scale to zero when idle, so an idle deploy costs nothing. To remove a
deployed app entirely:

```bash
modal app stop gemma4-26b-a4b-it-vllm
modal app stop gemma4-chat-ui
modal app stop gemma4-vision
modal app stop gemma4-live-camera
```

(Use `modal app list` to see running apps.) The `hf-cache` Volume persists across
deploys; delete it with `modal volume delete hf-cache` if you want to reclaim the
weight storage.
