# Third-Party Notices

This project's own source code is MIT-licensed (see `LICENSE`). The
libraries, models, and weights it uses at runtime carry their own,
separate licenses. This file summarizes them; it is not legal advice --
verify current terms on each project's own page before commercial use,
since model licenses in particular have changed across releases.

## Code dependencies (via `requirements.txt`)

| Package | License (at time of writing) |
|---|---|
| CrewAI | MIT |
| LangChain / langchain-core / langchain-ollama | MIT |
| LangGraph | MIT |
| Ollama (Python client) | MIT |
| Diffusers | Apache 2.0 |
| Transformers | Apache 2.0 |
| Accelerate | Apache 2.0 |
| PyTorch / torchvision | BSD-3-Clause |
| open-clip-torch | MIT |
| Pillow | MIT-CMU (a.k.a. HPND) |
| Pydantic / pydantic-settings | MIT |
| Streamlit | Apache 2.0 |
| ChromaDB | Apache 2.0 |
| pytest | MIT |

## Models (downloaded separately, not bundled in this repository)

- **Ollama LLMs** (default: `qwen2.5:7b`; alternative documented:
  `gemma3:4b`) -- each model has its own license/usage terms set by its
  publisher. Check the model's page on https://ollama.com/library before
  any commercial use; terms differ by model family and have changed
  across releases (e.g. Gemma has Google's Gemma Terms of Use; Qwen
  releases have used both Apache 2.0 and the Tongyi Qianwen license
  depending on version and size).
- **stabilityai/stable-diffusion-xl-base-1.0** (default image model) --
  CreativeML Open RAIL++-M License. This license permits commercial use
  but includes use-based restrictions (e.g. no generating unlawful
  content); read the full license on the model's Hugging Face page
  before commercial deployment.
- **CLIP weights** (`ViT-B-32` / `laion2b_s34b_b79k`, via open_clip) --
  MIT (open_clip) / the LAION-2B dataset the weights were trained on has
  its own usage terms; see the LAION and OpenCLIP project pages.
- **Optional aesthetic predictor weights** (not bundled; see README) --
  if you download LAION's aesthetic predictor weights separately, they
  carry LAION's own license terms -- check before use.

## What this project does NOT redistribute

No model weights are committed to or distributed by this repository.
`.env.example` and `requirements.txt` only reference where to obtain
models (Ollama's model library, Hugging Face Hub); the actual weight
files are downloaded by the user, under whatever terms the model
publisher sets, the first time each tool runs.

## A note on the original upstream repository

This project is a substantial rewrite/extension of
github.com/Methila-Meem/Instagram_Content_Creation_Multi-Agent_CrewAI.
As of inspection, that repository has no LICENSE file, which defaults to
"all rights reserved" under GitHub's Terms of Service -- meaning no
explicit permission was granted by its author to reuse, modify, or
redistribute its code. No source code from that repository was copied
verbatim into this one; its 4-agent CrewAI pipeline *concept* (research →
write → review → image-prompt) was used as a starting reference point and
reimplemented from scratch as part of a much larger system. If you are
the author of the original repository and would like attribution
language changed or added, please open an issue.
