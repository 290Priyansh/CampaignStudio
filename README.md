# CampaignStudio — Autonomous Multi-Agent Creative & Media Studio

A local-first, multi-agent generative AI studio that transforms a high-level campaign brief into a production-ready marketing campaign: creative strategy, audience profiling, multi-format asset planning, prompt optimization, local diffusion image generation, automated CLIP quality evaluation with closed-loop self-correction, and grounded social copywriting — all running 100% locally on your machine with **zero paid APIs**.

```text
"Launch a marketing campaign for an eco-friendly running shoe targeted at college students."
                                       │
                                       ▼
  CampaignStudio Production Package: Creative Strategy Brief, Target Audience Profile,
  Asset Breakdown, Quality-Gated SDXL Visuals, Social Copywriting, and Structured JSON Metadata
```

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Agent Workflow](#architecture--agent-workflow)
- [Tech Stack & Local AI Infrastructure](#tech-stack--local-ai-infrastructure)
- [No Paid APIs & Privacy Guardrails](#no-paid-apis--privacy-guardrails)
- [Dashboard & Demo UI Overview](#dashboard--demo-ui-overview)
- [Closed-Loop Evaluation & Self-Correction](#closed-loop-evaluation--self-correction)
- [Hardware Requirements & Performance Tuning](#hardware-requirements--performance-tuning)
- [Local Setup & Installation](#local-setup--installation)
- [Running CampaignStudio (UI & CLI)](#running-campaignstudio-ui--cli)
- [Campaign Deliverables & Output Structure](#campaign-deliverables--output-structure)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Docker Deployment](#docker-deployment)
- [Licensing](#licensing)
- [Resume Positioning](#resume-positioning)

---

## 🚀 Project Overview

Producing digital ad campaigns traditionally requires either paying per-generation for commercial cloud APIs or manually writing prompts, rendering images individually, eyeballing quality, and writing copy by hand. 

**CampaignStudio** automates this entire pipeline as a collaborative team of specialized AI agents running on local, open-source models:

1. **Creative Director Agent**: Synthesizes campaign brief inputs into a core creative strategy (visual style, color palette, typography guidelines, composition rules, photography direction).
2. **Audience Analyst Agent**: Researches target customer demographics, pain points, core motivations, emotional triggers, and social media scroll behavior.
3. **Campaign Strategist Agent**: Plans a cohesive set of ad deliverables (Hero Ad, Product Shot, Instagram Story, Lifestyle Shot) with targeted aspect ratios ($1:1, 4:5, 9:16$) and prompt specs.
4. **Prompt Optimizer Agent**: Automatically crafts text-to-image diffusion prompts (subject, environment, lighting, camera angle, negative prompt, guidance scale, steps).
5. **Local Diffusion Generator**: Renders high-resolution visual candidate images via PyTorch & Hugging Face `diffusers` (SDXL).
6. **Automated Vision Evaluator**: Scores candidate images against real, measurable criteria — CLIP semantic similarity (`ViT-B-32`) and technical quality checks (resolution, corruption, blank image, duplicates).
7. **Closed-Loop Self-Correction**: Candidates scoring below the **Quality Gate** trigger prompt revisions and automatic retries (hard-capped to prevent infinite loops).
8. **Copywriter Agent**: Writes headlines, body captions, calls-to-action (CTAs), and hashtags tailored specifically for each visual asset.
9. **Production Packaging**: Packages deliverables into an organized export directory (`outputs/<campaign_name>/`).

---

## 🏗️ Architecture & Agent Workflow

```text
                             Campaign Brief (Text Input)
                                          │
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │    Agent #1: Executive Creative Director    │
                   │ → CreativeBrief (objective, style, palette) │
                   └─────────────────────────────────────────────┘
                                          │
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │    Agent #2: Audience Analyst & Profiler    │
                   │ → AudienceProfile (demographics, tone, hooks)│
                   └─────────────────────────────────────────────┘
                                          │
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │     Agent #3: Multi-Format Strategist       │
                   │ → CampaignPlan (list of planned AssetPlans) │
                   └─────────────────────────────────────────────┘
                                          │
                                          ▼  (Iterated per planned asset)
                   ┌─────────────────────────────────────────────┐
                   │  Agent #4: Prompt Optimizer & Image Engine  │
                   │ → ImagePrompt & Diffusion Generation (SDXL) │
                   └─────────────────────────────────────────────┘
                                          │
                                          ▼
                   ┌─────────────────────────────────────────────┐
                   │     Automated Multimodal CLIP Evaluator     │
                   │ → Semantic CLIP Score & Technical Checks    │
                   └─────────────────────────────────────────────┘
                                          │
                               Score ≥ Quality Gate?
                               ┌──────────┴──────────┐
                              YES                    NO
                               │                     │
                               │                     ▼
                               │         Revise Prompt with Feedback
                               │         (Retry up to Max Attempts)
                               ▼                     │
                     Best Candidate Selected ────────┘
                               │
                               ▼
                   ┌─────────────────────────────────────────────┐
                   │         Agent #5: Creative Copywriter       │
                   │ → Headline, Caption, CTA, Hashtags per Asset │
                   └─────────────────────────────────────────────┘
                                          │
                                          ▼
                  outputs/<campaign_name>/
                  ├── campaign_brief.json    ├── strategy.json
                  ├── images/                ├── campaign_copy.json
                  └── evaluations/           └── README.md
```

---

## 🛠️ Tech Stack & Local AI Infrastructure

| Infrastructure Layer | Technology Choice | Rationale & Role |
| :--- | :--- | :--- |
| **Local LLM Serving** | **Ollama** | Zero-cost local inference server (`http://localhost:11434`). Swappable models via `.env`. |
| **Default LLM Model** | **`qwen2.5:7b`** (or `gemma3:4b`) | High reasoning performance for agent roleplay and schema generation. |
| **Agent Orchestration** | **CrewAI** | Multi-agent collaboration with role goals, backstories, and sequential task execution. |
| **Structured Output** | **LangChain** (`ChatOllama.with_structured_output`) | Guarantees strict JSON schema extraction into Pydantic models. |
| **Diffusion Image Engine** | **Hugging Face Diffusers** (`SDXL-base-1.0`) | Local, open-weight text-to-image diffusion model executing on CUDA GPUs. |
| **Semantic Vision Evaluator** | **OpenCLIP** (`ViT-B-32 / laion2b_s34b_b79k`) | Computes image-to-text cosine similarity embeddings locally. |
| **Data Contracts** | **Pydantic v2** | Enforces type validation across every agent and service boundary. |
| **User Interface** | **Streamlit** | Executive Studio Dashboard featuring glassmorphic styling and interactive demo views. |
| **Test Suite** | **pytest** | 85+ unit and integration tests executing without requiring network or GPU. |

---

## 🔒 No Paid APIs & Privacy Guardrails

- **Zero Cloud API Dependencies**: Does **not** require OpenAI, Anthropic, Midjourney, DALL-E, or Replicate API keys.
- **100% Private & Local**: All prompts, business briefs, target audience data, and generated images remain strictly on your local machine.
- **Offline Capable**: After downloading model weights on initial run, the system operates completely offline without internet connectivity.

---

## 🎬 Dashboard & Demo UI Overview

The **CampaignStudio** user interface ([ui/streamlit_app.py](file:///d:/ai-creative-director/ui/streamlit_app.py)) provides an executive studio dashboard organized into 9 structured tabs:

1. **🎬 Demo Showcase**: **Interactive Presentation View.** Displays each visual ad asset side-by-side with its strategic positioning, prompt optimization trail, CLIP evaluation metrics, and social deliverables.
2. **📊 Executive Summary**: Displays studio performance metrics: assets planned, quality gate pass rate, candidates evaluated, total attempts, and campaign export paths.
3. **🎯 Creative Brief**: **Agent #1 (Creative Director)** strategy brief: campaign objectives, key messaging, brand personality pills, visual style, photography direction, typography, and color palette.
4. **👥 Audience Insights**: **Agent #2 (Audience Analyst)** target profile: demographic summary, customer pain points, core motivations, emotional triggers, copy tone, and scroll behavior.
5. **📋 Campaign Plan**: **Agent #3 (Campaign Strategist)** asset breakdown: placement purposes, target audience slices, aspect ratios ($1:1, 4:5, 9:16$), and prompt specs.
6. **🎨 Generation & Loops**: **Agent #4 (Prompt Optimizer & Generator)** deep dive: initial prompts, candidate image evaluation scores, quality gate decisions, feedback received, and prompt revision history across attempts.
7. **✍️ Copywriting**: **Agent #5 (Copywriter)** social deliverables: headlines, post captions, CTAs, and hashtags.
8. **📜 Execution Logs**: Live, filterable agent execution log stream capturing raw LLM completion thoughts and tool calls.
9. **📦 Export Package**: Output directory locations and raw JSON viewers for `creative_brief.json`, `campaign_plan.json`, and `campaign_copy.json`.

---

## 🔬 Closed-Loop Evaluation & Self-Correction

Rather than relying on vague "LLM opinions" to determine whether an image is acceptable, [tools/image_evaluation.py](file:///d:/ai-creative-director/tools/image_evaluation.py) combines three deterministic metrics into an `EvaluationResult`:

1. **Semantic Similarity Score**: Uses OpenCLIP (`ViT-B-32`) to compute the cosine similarity between the positive diffusion prompt and the generated candidate image, normalized to $[0, 1]$.
2. **Technical Quality Score**: Performs non-ML deterministic checks: file validity, minimum resolution, near-blank/solid-color detection (pixel standard deviation), and exact perceptual duplicate checks against prior candidates.
3. **Optional Aesthetic Score**: Evaluates aesthetic appeal via local aesthetic predictor models when enabled.

### Self-Correction Loop
If the candidate image scores below `QUALITY_THRESHOLD` (e.g. `0.75`), the evaluation feedback is passed to the **Prompt Optimizer Agent**, which rewrites the diffusion prompt addressing specific issues (e.g., *"increase subject contrast"*, *"re-center subject"*) for attempt #2. Retries are hard-capped (`MAX_GENERATION_ATTEMPTS`) to prevent infinite loops.

---

## ⚡ Hardware Requirements & Performance Tuning

### Hardware Guidelines

| Component | Minimum Requirements | Recommended Setup |
| :--- | :--- | :--- |
| **RAM** | 16 GB | 32 GB |
| **Ollama LLM Model** | `gemma3:4b` (~3 GB VRAM/RAM) | `qwen2.5:7b` (~5 GB VRAM/RAM) |
| **GPU (Image Gen)** | CUDA GPU with 4GB+ VRAM (e.g. RTX 3050) | NVIDIA CUDA GPU with 8GB+ VRAM (RTX 3060/4060+) |
| **Disk Space** | ~15 GB free (models + outputs) | 30+ GB |

### Performance Optimization Tips (5x–10x Faster Execution)

If running on laptops with mid-range GPUs (e.g., NVIDIA RTX 3050 Laptop GPU):
1. **Reduce Assets**: Set "Number of Assets" slider in the sidebar to `1` or `2` for quick iterations.
2. **Reduce Diffusion Steps**: In `.env`, set `IMAGE_NUM_INFERENCE_STEPS=12` or `15` (instead of 20).
3. **Lower Quality Threshold**: Set "Quality Threshold" to `0.65` or `0.70` to pass images on attempt #1 without triggering retry loops.
4. **Use Lightweight LLMs**: Set `OLLAMA_MODEL=gemma3:4b` or `qwen2.5:3b`.

---

## 📥 Local Setup & Installation

```bash
# 1. Clone the repository and enter directory
git clone <repository-url>
cd ai-creative-director

# 2. Set up Python virtual environment
python3 -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install & Start Ollama (https://ollama.com)
ollama serve
ollama pull qwen2.5:7b

# 5. Configure environment settings
cp .env.example .env

# 6. Run test suite (verifies setup without requiring GPU or live server)
pytest
```

---

## 🏃 Running CampaignStudio (UI & CLI)

### Launch Web Dashboard (Streamlit UI)
```bash
streamlit run ui/streamlit_app.py
```

### Launch Command Line Interface (CLI)
```bash
python -m app.main "Launch a marketing campaign for an eco-friendly running shoe targeted at college students." --assets 3
```

---

## 📁 Campaign Deliverables & Output Structure

Running a campaign exports a complete production package into `outputs/<campaign_name>/`:

```text
outputs/ecostride_launch/
├── campaign_brief.json      # Structured CreativeBrief + AudienceProfile JSON
├── campaign_plan.json       # Asset plans & aspect ratio specifications
├── images/
│   ├── hero_01.png          # Selected quality-gated final visual asset
│   ├── lifestyle_02.png
│   └── story_03.png
├── evaluations/
│   ├── hero_01.json         # Full evaluation result (semantic, technical, overall scores)
│   ├── lifestyle_02.json
│   └── story_03.json
├── campaign_copy.json       # Headlines, captions, CTAs, and hashtags per asset
└── README.md                # Human-readable markdown summary report
```

---

## 🧪 Testing & Quality Assurance

CampaignStudio includes a test suite with 85+ tests covering models, schemas, agent tasks, evaluation metrics, image generation control flow, and UI rendering:

```bash
# Run full unit and integration test suite
pytest
```

- **Mocked Dependencies**: PyTorch, Diffusers, CrewAI, and Ollama are mocked at API boundaries, allowing tests to run rapidly in standard CI environments without GPU hardware.
- **UI Testing**: The Streamlit interface is smoke-tested using Streamlit's native `AppTest` harness.

---

## 🐳 Docker Deployment

```bash
docker compose up --build
```

- The container executes the application environment while referencing Ollama on the host machine (`OLLAMA_BASE_URL=http://host.docker.internal:11434`).
- For GPU acceleration in Docker, install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) and enable the GPU block in `docker-compose.yml`.

---

## 📄 Licensing

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Third-party dependency licenses and model weight terms are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 💼 Resume Positioning

**CampaignStudio — Autonomous Multi-Agent Creative & Media Studio**
- Architected a 100% local multi-agent generative AI system (CrewAI + LangChain over Ollama) that converts natural-language briefs into structured brand positioning, target audience profiles, multi-format ad plans, diffusion prompts, and social media copy.
- Developed an automated visual quality gate combining OpenCLIP semantic embedding similarity (`ViT-B-32`) and technical image checks into a closed-loop generate $\rightarrow$ evaluate $\rightarrow$ revise $\rightarrow$ retry optimization workflow.
- Built a modular Python architecture using Pydantic v2 data contracts across agent boundaries, an executive Streamlit dashboard with interactive demo views, a CLI entry point, and an 85+ test suite running in CI environments without GPU dependencies.
