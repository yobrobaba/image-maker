# diffusion-colab-t4

Run Stable Diffusion 3.5 (txt2img, img2img, inpainting) on a free-tier
Google Colab T4 GPU, using 4-bit (NF4) quantization to fit in 16GB VRAM.

## Setup

1. Open `notebook.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Upload `config.yaml` and `requirements.txt` into the Colab session
   (or clone this repo directly inside Colab — see notebook cell 1).
4. Get a Hugging Face token (read-only is fine) and accept the license
   on the model page: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
   Add the token to Colab under **Secrets** (key icon in the left sidebar)
   as `HF_TOKEN`. Do not paste tokens directly into cells.
5. Run all cells top to bottom.

## Swapping models

Everything model-specific lives in `config.yaml`. To try a different
SD3-family checkpoint, change `model.id`. No notebook code needs to change.

**About FLUX**: FLUX.1-dev/schnell work for txt2img with the same
quantization pattern, but the inpainting-capable variant (FLUX.1-Fill-dev)
is separately gated and non-commercial licensed, and diffusers' FLUX
pipelines don't share the exact same `AutoPipelineFor*` ergonomics used
here. If you want to switch to FLUX, ping me — the pipeline-loading cell
needs a small rewrite, not just a config change.

## Memory notes (T4 = 16GB)

- `enable_model_cpu_offload()` keeps only active submodules on GPU —
  required, not optional, at this VRAM budget.
- NF4 quantization (via `bitsandbytes`) roughly quarters the transformer's
  memory footprint versus fp16.
- The notebook loads the model **once** and derives img2img/inpaint
  pipelines from it via `from_pipe()`, so you don't triple your VRAM usage
  by loading three separate pipelines.
- If you hit OOM: lower `height`/`width` in `config.yaml` (768 instead of
  1024), or restart the runtime — fragmented VRAM from a crashed
  generation won't free itself otherwise.

## Repo structure

```
diffusion-colab-t4/
├── notebook.ipynb     # main entry point, run in Colab
├── config.yaml         # model + generation settings
├── requirements.txt
├── outputs/            # generated images (gitignored)
└── README.md
```
