# VisionForge — AI Creative Director

A local-first, multi-agent generative AI system that takes a plain-English
campaign brief and produces a full Instagram campaign: creative strategy,
optimized image-generation prompts, AI-generated visuals, automated quality
evaluation with regeneration of weak results, and grounded social copy —
all running on your own machine, with no paid API required.

```
"Create a marketing campaign for an eco-friendly running shoe
 targeted at college students aged 18-24."
        │
        ▼
  a working campaign package: creative brief, campaign plan,
  generated + evaluated images, and Instagram-ready copy
```

This is a substantial rewrite/extension of
[`Methila-Meem/Instagram_Content_Creation_Multi-Agent_CrewAI`](https://github.com/Methila-Meem/Instagram_Content_Creation_Multi-Agent_CrewAI),
a 4-agent CrewAI notebook that generated Instagram captions and a single
Stable Diffusion image from a topic string. See
["What was inherited vs. newly built"](#what-was-inherited-vs-newly-built)
below for exactly what changed.

---

## Table of contents

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [No paid APIs](#no-paid-apis)
- [Hardware requirements](#hardware-requirements)
- [Local setup](#local-setup)
- [Running it](#running-it)
- [Example](#example)
- [Agent architecture](#agent-architecture)
- [Evaluation](#evaluation)
- [Regeneration loop](#regeneration-loop)
- [What was inherited vs. newly built](#what-was-inherited-vs-newly-built)
- [Testing](#testing)
- [Docker](#docker)
- [Licensing](#licensing)
- [Resume positioning](#resume-positioning)
- [Future improvements](#future-improvements)

---

## Project overview

Producing on-brand social content today usually means either paying for a
hosted AI API per generation, or manually writing prompts, generating
images one at a time, eyeballing whether they're any good, and writing
captions by hand. VisionForge automates that whole loop as a small team of
cooperating agents, entirely on local, open-source models:

1. A **Creative Director** turns a one-paragraph brief into a structured
   creative direction (visual style, color, tone, number of assets).
2. An **Audience Analyst** turns "college students" into concrete
   recommendations that actually change what gets produced.
3. A **Campaign Strategist** breaks the brief into a concrete list of
   assets (hero image, lifestyle shot, poster, story, etc.).
4. For each asset, a **Prompt Optimizer** writes a diffusion-ready prompt
   (subject, lighting, camera, composition, negative prompt — not just
   "campaign description → Stable Diffusion").
5. A local diffusion model generates several **candidate images**.
6. An **evaluator** scores each candidate on real, measurable criteria —
   CLIP image↔prompt similarity, optional aesthetic scoring, and
   deterministic technical checks (resolution, blank-image detection,
   duplicates) — not "ask the LLM if it looks good."
7. Candidates that fail the quality threshold trigger an automatic
   **prompt revision and retry**, capped at a fixed number of attempts.
8. A **Copywriter** writes Instagram copy grounded in the actual selected
   image and its role in the campaign — not generic filler.
9. Everything is packaged into a clean, inspectable output folder.

## Architecture

```
Campaign brief (plain text)
        │
        ▼
 ┌─────────────────┐
 │ Creative Director│  → CreativeBrief (visual style, tone, color, #assets)
 └─────────────────┘
        │
        ▼
 ┌─────────────────┐
 │ Audience Analyst │  → AudienceProfile (attached to the brief)
 └─────────────────┘
        │
        ▼
 ┌──────────────────┐
 │Campaign Strategist│ → CampaignPlan (a list of concrete AssetPlans)
 └──────────────────┘
        │
        ▼           (repeated for every planned asset)
 ┌────────────────┐
 │ Prompt Optimizer│  → ImagePrompt (positive/negative/lighting/camera/style)
 └────────────────┘
        │
        ▼
 ┌────────────────┐
 │ Image Generator │  → N candidate images (local diffusion)
 └────────────────┘
        │
        ▼
 ┌────────────────┐
 │    Evaluator    │  → EvaluationResult per candidate (CLIP + aesthetic + technical)
 └────────────────┘
        │
    score ≥ threshold?
    ┌────┴────┐
   YES        NO ──► revise prompt with evaluator feedback ──► retry
    │                (capped at MAX_GENERATION_ATTEMPTS)
    ▼
 Best candidate selected
        │
        ▼
 ┌────────────┐
 │ Copywriter │  → headline / caption / CTA / hashtags, grounded in the
 └────────────┘     actual selected image and its role in the campaign
        │
        ▼
 outputs/<campaign_slug>/
   campaign_brief.json, strategy.json, prompts/, images/, evaluations/,
   campaign_copy.json
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Local LLM serving | **Ollama** | Zero-cost local inference; swappable model via `.env` |
| Default model | **qwen2.5:7b** | Strong reasoning at a size most consumer GPUs/CPUs can run |
| Agent orchestration | **CrewAI** | Role/goal/backstory agents, task delegation, `Process.sequential` crews for the Creative Director / Audience Analyst / Campaign Strategist / Copywriter |
| Guaranteed-schema extraction | **LangChain** (`ChatOllama` + `with_structured_output`) | The Prompt Optimizer's job is a narrow, mechanical transformation, not persona role-play — LangChain's structured output is the more direct tool here, and it's what replaces the original repo's regex-scraped JSON with an actually-validated Pydantic object |
| Image generation | **Hugging Face Diffusers** (SDXL-base-1.0 by default) | Fully local, open-weight, swappable via `.env`; abstracted behind an `ImageGenerator` interface (`ComfyUIGenerator` also scaffolded) |
| Semantic evaluation | **open_clip** (ViT-B-32) | Real, measurable image↔prompt similarity, not an LLM opinion |
| Data contracts | **Pydantic v2** | Every agent/tool boundary in this project passes a validated model, not a raw string |
| UI | **Streamlit** | Thin presentation layer; the agent workflow is the actual project, per design intent |
| Tests | **pytest** | 80+ tests, no GPU or live Ollama server required (see [Testing](#testing)) |

### CrewAI vs. LangChain — why both, and what each owns

CrewAI drives the four **role-played** agents (Creative Director, Audience
Analyst, Campaign Strategist, Copywriter) — each has a persona backstory
and a task description, and CrewAI's `Task(output_pydantic=...)` gives
schema-shaped output for that role-play pattern. The **Prompt Optimizer**
is different: it isn't playing a character, it's mechanically transforming
a creative brief into diffusion parameters, so it goes straight through
LangChain's `ChatOllama.with_structured_output()`
(`models/llm.py::invoke_structured`), with explicit retry-on-malformed-
output handling. Both point at the exact same local Ollama server — there
is one model-serving process either way, just two different call patterns
for two different kinds of agent work. `ORCHESTRATION_FRAMEWORK=langgraph`
is present in settings as a documented extension point if you want to
migrate the top-level sequencing to LangGraph, but the current
implementation only wires up CrewAI end-to-end — see
[Future improvements](#future-improvements).

## No paid APIs

**The default configuration runs entirely locally and does not require
OpenAI, Anthropic, Gemini, Replicate, or any other paid inference API.**
Every LLM call goes to your local Ollama server; every image is generated
by a local Diffusers pipeline (or your own local ComfyUI instance). A
`.env`-driven settings guardrail (`config/settings.py`) is unit-tested to
confirm no OpenAI/Anthropic/Gemini/Replicate API-key field exists on the
settings object at all.

The one external network dependency is the **first-time download** of
model weights (from Ollama's library and/or Hugging Face Hub) — after
that, everything runs offline.

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| Ollama model | `gemma3:4b` (~3 GB) | `qwen2.5:7b` (~5 GB) — the project default |
| GPU (for image generation) | None (CPU works, but see caveat below) | NVIDIA GPU, 8+ GB VRAM |
| VRAM (SDXL-base-1.0, fp16) | — | ~8 GB |
| Disk | ~15 GB free (models + generated outputs) | 30+ GB |

**Be realistic about CPU-only image generation**: SDXL on CPU is not a
"slightly slower" experience — expect several minutes per image, and a
multi-candidate, multi-attempt campaign run will take a long time. The
LLM/agent side of this project (`OLLAMA_*` settings) runs fine on CPU. The
diffusion side (`IMAGE_DEVICE`) genuinely benefits from a CUDA GPU. If you
don't have one:

- Set `IMAGE_DEVICE=cpu` and expect long generation times, or
- Lower `IMAGE_NUM_INFERENCE_STEPS` and `CANDIDATES_PER_ASSET` in `.env`
  to reduce total work, or
- Use a smaller/faster base model (e.g. an SD1.5 checkpoint instead of
  SDXL) via `IMAGE_MODEL`.

## Local setup

```bash
# 1. Clone this repository and enter it
git clone <this-repo-url>
cd ai-creative-director

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Install and start Ollama (https://ollama.com)
ollama serve &
ollama pull qwen2.5:7b
# Lower-resource alternative:
#   ollama pull gemma3:4b   (then set OLLAMA_MODEL=gemma3:4b in .env)

# 4. Configure
cp .env.example .env
# Edit .env if your hardware needs different defaults (see table above).
# The first real image generation will download SDXL-base-1.0 from
# Hugging Face (~7 GB) -- this happens automatically on first use.

# 5. Run the test suite (no GPU/Ollama needed for this)
pytest
```

## Running it

**Streamlit UI:**

```bash
streamlit run ui/streamlit_app.py
```

**CLI:**

```bash
python -m app.main "Create a marketing campaign for an eco-friendly running shoe targeted at college students aged 18-24." --assets 3
```

Both call the same `services.campaign_service.run_full_campaign()` entry
point — the UI is presentation only, per the project's design intent.

## Example

Input brief:

> "Create a marketing campaign for an eco-friendly running shoe targeted
> at college students aged 18-24. The brand should feel energetic, modern
> and environmentally conscious."

Output (`outputs/ecostride_launch/`):

```
ecostride_launch/
├── campaign_brief.json      # full structured CreativeBrief + AudienceProfile
├── strategy.json            # CampaignPlan: list of planned assets
├── prompts/
│   ├── hero.txt             # positive/negative prompt actually used
│   ├── lifestyle.txt
│   └── poster.txt
├── images/
│   ├── hero.png             # the selected, quality-gated final image
│   ├── lifestyle.png
│   └── poster.png
├── evaluations/
│   ├── hero.json            # {semantic_score, aesthetic_score, technical_score, overall_score, passed, feedback}
│   ├── lifestyle.json
│   └── poster.json
└── campaign_copy.json       # headline/caption/CTA/hashtags per asset
```

## Agent architecture

| Agent | Framework | Input | Output |
|---|---|---|---|
| **Creative Director** (`agents/creative_director.py`) | CrewAI | raw brief text | `CreativeBrief` |
| **Audience Analyst** (`agents/audience_analyst.py`) | CrewAI | `CreativeBrief` | `AudienceProfile` (attached to the brief) |
| **Campaign Strategist** (`agents/campaign_strategist.py`) | CrewAI | `CreativeBrief` (+ audience profile) | `CampaignPlan` (list of `AssetPlan`) |
| **Prompt Optimizer** (`agents/prompt_optimizer.py`) | LangChain structured output | `CreativeBrief` + `AssetPlan` (+ feedback, on retry) | `ImagePrompt` |
| **Copywriter** (`agents/copywriter.py`) | CrewAI | `CreativeBrief` + selected `GeneratedAsset`s | `CampaignCopy` |

Every agent boundary is a validated Pydantic model (`schemas/`) — nothing
in this pipeline passes a raw string between stages and hopes for the
best, which was the original repo's approach (regex-scraping a `\`\`\`json`
fence out of free-form LLM output).

## Evaluation

`evaluation/quality_gate.py` combines three real, independently-testable
signals into one `EvaluationResult`:

- **Semantic score** (`evaluation/clip_evaluator.py`) — CLIP cosine
  similarity between the generated image and its own prompt (ViT-B-32,
  `laion2b_s34b_b79k`), rescaled to `[0, 1]`.
- **Technical score** (`evaluation/technical_checks.py`) — deterministic,
  no-ML checks: file validity, minimum resolution, near-blank/solid-color
  detection (pixel standard deviation), and exact-duplicate detection
  against every candidate generated so far in the run.
- **Aesthetic score** (`evaluation/aesthetic_evaluator.py`) — *optional*.
  A LAION-Aesthetics-style MLP on top of frozen CLIP embeddings. Weights
  aren't bundled with this repo (a separate ~5 MB download); if you don't
  configure a weights path, this returns `None` and its weight (0.2) is
  redistributed across semantic (0.5) and technical (0.3) rather than
  silently capping your maximum achievable score.

This directly replaces the "ask the LLM if the image looks good"
anti-pattern with numbers you can log, threshold, and reason about.

## Regeneration loop

`workflows/asset_generation.py::generate_asset()` implements the loop:

1. Prompt Optimizer produces an initial `ImagePrompt`.
2. `CANDIDATES_PER_ASSET` images are generated from it and each is scored.
3. If the best candidate passes `QUALITY_THRESHOLD` → done.
4. Otherwise, the Prompt Optimizer revises the prompt using the failing
   candidate's `feedback` list, and generation retries.
5. Hard-capped at `MAX_GENERATION_ATTEMPTS` (default 3) — **never** an
   infinite loop. If every attempt fails the threshold, the best-scoring
   candidate across all attempts is still returned, explicitly marked
   `passed=False`, so a demo never hangs or silently produces nothing.

Both limits are `.env`-configurable:

```env
QUALITY_THRESHOLD=0.75
MAX_GENERATION_ATTEMPTS=3
CANDIDATES_PER_ASSET=3
```

## What was inherited vs. newly built

| From the original repo | Status |
|---|---|
| 4-agent pipeline *concept* (research → write → review → image-prompt) | Reimplemented from scratch as the Creative Director / Audience Analyst / Campaign Strategist / Copywriter split — no code copied verbatim |
| CrewAI as the orchestration framework | Retained and extended (real `output_pydantic` schemas instead of regex-scraped JSON) |
| Ollama as the local LLM backend | Retained; model upgraded from `gemma3:4b` to configurable, defaulting to `qwen2.5:7b` |
| Local Diffusers image generation | Retained the *approach*; replaced the hardcoded inline call and the non-commercially-licensed `dreamlike-photoreal-2.0` checkpoint with an abstracted `ImageGenerator` interface defaulting to openly-licensed SDXL-base-1.0 |
| `langchain_ollama`/`langchain_community` in requirements | Were unused in the original notebook (zero imports); this project actually uses LangChain for structured output extraction |
| Colab notebook, `userdata` HF token, single `.ipynb` file | Removed entirely — this is now a modular Python package with no Colab dependency |

**Newly built, not present in the original at all:** Prompt Optimizer
agent, CLIP/technical/aesthetic evaluation, the generate→evaluate→revise→
retry loop, multi-candidate generation and ranking, structured Pydantic
schemas throughout, the `ImageGenerator`/`ImageEvaluator` abstractions,
campaign output packaging, the Streamlit UI, the CLI, logging, and the
full test suite.

## Testing

```bash
pytest            # ~80+ tests, a few seconds, no GPU or network required
```

External dependencies are mocked at their boundaries rather than the test
suite requiring a live Ollama server or GPU:

- `torch`/`diffusers` are injected as fake modules (`sys.modules`) to test
  `DiffusersGenerator`'s own control flow (paths, seeding, error mapping)
  without a real model download.
- `crewai` is faked the same way (see `tests/conftest.py::fake_crewai`) to
  test each agent's task construction and error handling.
- `evaluation/technical_checks.py` is tested against **real** tiny images
  generated with Pillow (these checks are cheap/deterministic — mocking
  them would just test the mock).
- The Streamlit UI is smoke-tested with Streamlit's own
  `streamlit.testing.v1.AppTest` harness, which actually executes the
  script and drives its widgets.

## Docker

```bash
docker compose up --build
```

`Dockerfile`/`docker-compose.yml` package the **Python application only**.
Docker does not solve GPU access or bundle Ollama for you:

- The container talks to Ollama running on your **host** machine
  (`OLLAMA_BASE_URL=http://host.docker.internal:11434`), rather than
  running a second Ollama instance in Docker — simpler, and avoids a
  duplicate model cache.
- For GPU-accelerated generation inside the container, install the
  [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  on the host and uncomment the `deploy.resources` GPU block in
  `docker-compose.yml`. Without it, set `IMAGE_DEVICE=cpu` in `.env` and
  expect the CPU timings discussed above.

## Licensing

This project's own source code is MIT-licensed (`LICENSE`). Runtime
dependencies and downloaded model weights carry their own, separate
licenses — see `THIRD_PARTY_NOTICES.md` for a full breakdown (Ollama
model licenses vary by model/version; SDXL-base-1.0 is CreativeML Open
RAIL++-M; CLIP/open_clip weights are MIT with LAION-2B dataset terms).
`THIRD_PARTY_NOTICES.md` also documents this project's relationship to
the original upstream repository, which carries no LICENSE file.

## Resume positioning

**VisionForge — Local Multi-Agent AI Creative Director**

- Designed and implemented a local, multi-agent generative AI pipeline
  (CrewAI + LangChain structured output over Ollama) that decomposes a
  natural-language campaign brief into a validated creative strategy,
  asset plan, and per-asset diffusion prompts — with zero paid inference
  API dependency.
- Built an automated visual quality gate combining CLIP image-text
  similarity, deterministic technical checks, and optional aesthetic
  scoring into a single measurable threshold, driving an attempt-capped
  generate→evaluate→revise→retry loop with multi-candidate ranking.
- Refactored a single-notebook prototype into a modular, dependency-
  injected Python package (Pydantic schemas at every agent/tool boundary,
  swappable local LLM and image-generation backends) with an 80+ test
  suite requiring no GPU or live model server, a Streamlit UI, a CLI, and
  Docker packaging.

*(These claims are scoped to match what's actually implemented and tested
in this repository — no unmeasured metrics are claimed.)*

## Future improvements

- Migrate the top-level sequencing (`workflows/campaign_workflow.py`) to
  LangGraph for explicit state-machine visualization, now that
  `ORCHESTRATION_FRAMEWORK` exists as a documented switch.
- Finish `ComfyUIGenerator`'s result-polling (currently submits a workflow
  but doesn't consume its output — see the class docstring for why this
  is workflow-specific and out of scope for a generic implementation).
- Optional local brand-knowledge RAG (Ollama embeddings + Chroma) so the
  Creative Director can ground campaigns in retrieved brand guidelines —
  scaffolded as an opt-in setting (`RAG_ENABLED`) but not yet implemented,
  per the brief's instruction not to add RAG just to add RAG.
- Basic image-embedding-based style/character consistency across assets
  in the same campaign (simple CLIP-similarity check between an asset and
  a chosen reference, rather than a full IP-Adapter/ControlNet setup).
