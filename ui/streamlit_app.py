"""
CampaignStudio — Multi-Agent Creative & Media Interface

A modern studio interface:
  Creative Director -> Audience Analyst -> Campaign Strategist
  -> Prompt Optimization -> Local Diffusion Generation -> CLIP Evaluation -> Copywriting.

Everything runs 100% locally.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config import settings
from models.exceptions import CreativeDirectorError
from models.llm import check_ollama_available
from services.campaign_service import run_full_campaign


class StreamlitLogHandler(logging.Handler):
    """Collects formatted log records into a list for display after the run."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
        except Exception:
            try:
                self.records.append(str(record.getMessage()))
            except Exception:
                self.records.append(str(record.msg))


def apply_ui_overrides(ollama_model: str, quality_threshold: float) -> None:
    """Mutate settings singleton for this execution session."""
    settings.ollama_model = ollama_model
    settings.quality_threshold = quality_threshold


def build_full_brief_text(campaign_brief: str, target_audience: str, brand_style: str) -> str:
    """Combine brief inputs into the single prompt consumed by the Creative Director agent."""
    text = campaign_brief
    if target_audience:
        text += f"\n\nTarget audience: {target_audience}"
    if brand_style:
        text += f"\nBrand style/personality: {brand_style}"
    return text


def inject_custom_css() -> None:
    """Inject custom typography and remove generic AI box wrappers."""
    st.markdown(
        """
        <style>
        /* Base typography & dark theme */
        .stApp {
            background-color: #0b0d12;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* Hide Streamlit Deploy button & header deploy options */
        .stAppDeployButton,
        [data-testid="stAppDeployButton"],
        header [data-testid="stAppDeployButton"],
        button[title="Deploy"] {
            display: none !important;
        }

        /* Remove dark blue bulky AI container boxes */
        div[data-testid="stNotification"] {
            background-color: transparent !important;
            border: none !important;
            border-left: 3px solid #3b82f6 !important;
            border-radius: 0px !important;
            padding: 8px 16px !important;
            color: #94a3b8 !important;
        }

        .studio-title-main {
            font-size: 2.4rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
            margin-bottom: 2px;
        }

        .studio-tagline {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-bottom: 28px;
            font-weight: 400;
        }

        .section-category {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #3b82f6;
            margin-bottom: 6px;
        }

        .agent-thinking-card {
            background-color: #131722;
            border-left: 3px solid #3b82f6;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
        }

        .agent-thinking-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #60a5fa;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .agent-thinking-text {
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.5;
        }

        .tag-pill {
            background-color: #1e293b;
            color: #93c5fd;
            border: 1px solid #334155;
            font-size: 0.8rem;
            font-weight: 500;
            padding: 3px 10px;
            border-radius: 14px;
            display: inline-block;
            margin-right: 6px;
            margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_asset(asset, asset_copy) -> None:
    """Render legacy asset view helper for compatibility."""
    status = "passed" if asset.evaluation.passed else "did not pass threshold"
    st.subheader(f"{asset.asset_id} — score {asset.evaluation.overall_score:.2f} ({status})")

    left, right = st.columns([1, 1])
    with left:
        st.image(asset.file_path, caption="Selected final image", use_container_width=True)
        st.text_area(f"Prompt used ({asset.asset_id})", asset.prompt.positive_prompt, height=80)

    with right:
        if asset_copy is not None:
            st.markdown(f"**{asset_copy.headline}**")
            st.write(asset_copy.caption)
            st.write(asset_copy.cta)
            st.write(" ".join(f"#{h.lstrip('#')}" for h in asset_copy.hashtags))

        st.metric("Semantic score", f"{asset.evaluation.semantic_score:.2f}")
        st.metric("Technical score", f"{asset.evaluation.technical_score:.2f}")
        if asset.evaluation.aesthetic_score is not None:
            st.metric("Aesthetic score", f"{asset.evaluation.aesthetic_score:.2f}")
        st.metric("Attempts used", asset.attempts_used)


def render_demo_showcase(result, copy) -> None:
    """Renders visual deliverables alongside agent decisions and thinking rationale."""
    st.markdown("<div class='section-category'>Studio Demo Presentation</div>", unsafe_allow_html=True)
    st.markdown("### 🎬 Visual Assets & Agent Strategic Decisions")
    st.caption("Side-by-side showcase linking visual ad assets directly to agent rationale, prompt optimization, and copy outputs.")

    copy_by_asset = {c.asset_id: c for c in copy.assets}
    plan_by_asset = {p.asset_id: p for p in result.plan.assets}

    for idx, asset in enumerate(result.assets, 1):
        asset_copy = copy_by_asset.get(asset.asset_id)
        asset_plan = plan_by_asset.get(asset.asset_id)

        with st.container():
            st.markdown(f"#### Asset #{idx}: `{asset.asset_id}`")
            col_img, col_details = st.columns([1.1, 1.3])

            with col_img:
                st.image(asset.file_path, caption=f"Selected Final Output ({asset.asset_id}.png)", use_container_width=True)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall Score", f"{asset.evaluation.overall_score:.2f}")
                m2.metric("CLIP Similarity", f"{asset.evaluation.semantic_score:.2f}")
                m3.metric("Attempts", f"{asset.attempts_used}")

            with col_details:
                # Strategic decision rationale
                if asset_plan:
                    st.markdown("##### 💡 Strategic Placement & Intent")
                    st.write(f"**Asset Type:** `{asset_plan.asset_type.value}`")
                    st.write(f"**Purpose:** {asset_plan.purpose}")
                    st.write(f"**Message:** {asset_plan.message}")
                    st.write(f"**Visual Direction:** {asset_plan.visual_direction}")

                # Generation & Optimization trail
                st.markdown("##### 🧠 Prompt Optimization & Generation Loop")
                st.write(f"**Final Prompt Used:** `{asset.prompt.positive_prompt}`")
                if asset.prompt.negative_prompt:
                    st.caption(f"**Negative Prompt:** `{asset.prompt.negative_prompt}`")

                if asset.attempts_used > 1:
                    st.warning(
                        f"🔄 **Optimization Cycle Applied:** Asset required {asset.attempts_used} attempts. "
                        f"Evaluator feedback addressed: {'; '.join(asset.evaluation.feedback or ['Prompt alignment improved'])}"
                    )
                else:
                    st.success("✅ **First-Attempt Pass:** Met quality gate threshold on attempt #1.")

                # Social Media Deliverables
                if asset_copy:
                    st.markdown("##### ✍️ Final Social Copy Deliverables")
                    st.markdown(f"**Headline:** `{asset_copy.headline}`")
                    st.write(f"**Caption:** {asset_copy.caption}")
                    st.write(f"**Call To Action:** `{asset_copy.cta}`")
                    tags_html = "".join(f"<span class='tag-pill'>#{h.lstrip('#')}</span>" for h in asset_copy.hashtags)
                    st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("---")


def render_executive_summary(result, copy, campaign_dir) -> None:
    st.markdown("<div class='section-category'>Studio Metrics</div>", unsafe_allow_html=True)
    st.markdown("### 📊 Executive Summary")
    
    c1, c2, c3, c4 = st.columns(4)
    passed_count = sum(1 for a in result.assets if a.evaluation.passed)
    total_assets = len(result.assets)
    total_candidates = sum(len(a.all_candidates) for a in result.assets)
    total_attempts = sum(a.attempts_used for a in result.assets)

    c1.metric("Assets Planned", total_assets)
    c2.metric("Passed Quality Gate", f"{passed_count}/{total_assets}")
    c3.metric("Candidates Evaluated", total_candidates)
    c4.metric("Total Generation Attempts", total_attempts)

    st.write(f"📁 **Campaign Package Directory:** `{campaign_dir}`")
    st.markdown("---")

    st.markdown("#### 🖼️ Final Deliverables Showcase Grid")
    cols = st.columns(min(total_assets, 4))
    copy_by_asset = {c.asset_id: c for c in copy.assets}
    for idx, asset in enumerate(result.assets):
        col = cols[idx % 4]
        with col:
            st.image(asset.file_path, use_container_width=True)
            status_badge = "✅ PASSED" if asset.evaluation.passed else "⚠️ RETRIED / BELOW GATE"
            st.markdown(f"**{asset.asset_id}** ({status_badge})")
            st.caption(f"Score: `{asset.evaluation.overall_score:.2f}` | Attempts: `{asset.attempts_used}`")
            ac = copy_by_asset.get(asset.asset_id)
            if ac:
                st.markdown(f"*{ac.headline}*")


def render_creative_brief(brief) -> None:
    st.markdown("<div class='section-category'>Agent #1: Executive Creative Director</div>", unsafe_allow_html=True)
    st.markdown("### 🎯 Creative Strategy & Positioning Brief")

    st.markdown(
        """
        <div class='agent-thinking-card'>
            <div class='agent-thinking-title'>🧠 Agent Thinking & Rationale</div>
            <div class='agent-thinking-text'>
                Synthesized raw input into a core campaign objective, brand positioning, artistic rules, color palette, and photography style to anchor all downstream agents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"**Campaign Name:** `{brief.campaign_name}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📌 Brand Positioning & Messaging")
        st.write(f"**Objective:** {brief.campaign_objective}")
        st.write(f"**Key Message:** {brief.key_message}")
        st.write(f"**Target Audience:** {brief.target_audience}")

        st.markdown("#### 🏷️ Brand Personality Attributes")
        pills_html = "".join(f"<span class='tag-pill'>{p}</span>" for p in brief.brand_personality)
        st.markdown(pills_html, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🎨 Art & Aesthetic Direction")
        st.write(f"**Visual Style:** {brief.visual_style}")
        st.write(f"**Photography Style:** {brief.photography_style}")
        st.write(f"**Typography Guidelines:** {brief.typography_direction}")
        st.write(f"**Composition Rules:** {brief.composition_guidelines}")

        st.markdown("#### 🎨 Palette Direction")
        color_pills = "".join(f"<span class='tag-pill'>{c}</span>" for c in brief.color_direction)
        st.markdown(color_pills, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💡 Strategy & Content Ideas")
    st.write(f"**Prompt Strategy Notes:** {brief.image_generation_strategy}")
    st.markdown("**Content Ideas:**")
    for idea in brief.content_ideas:
        st.markdown(f"- {idea}")


def render_audience_insights(brief) -> None:
    st.markdown("<div class='section-category'>Agent #2: Consumer Insights & Audience Analyst</div>", unsafe_allow_html=True)
    st.markdown("### 👥 Target Audience Profile & Psychological Insights")

    profile = brief.audience_profile
    if not profile:
        st.warning("No Audience Profile populated in brief.")
        return

    st.markdown(
        """
        <div class='agent-thinking-card'>
            <div class='agent-thinking-title'>🧠 Agent Thinking & Rationale</div>
            <div class='agent-thinking-text'>
                Profiled demographic segments, customer pain points, core motivations, and social media scroll habits to craft the strategic messaging hook and copy tone.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(f"**Demographic Summary:** {profile.demographic_summary}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Core Motivations")
        for m in profile.motivations:
            st.markdown(f"- {m}")

        st.markdown("#### ⚠️ Customer Pain Points")
        for p in profile.pain_points:
            st.markdown(f"- {p}")

    with col2:
        st.markdown("#### 🎭 Emotional & Behavioral Profile")
        st.write(f"**Desired Emotion:** {profile.desired_emotional_response}")
        st.write(f"**Recommended Copy Tone:** {profile.recommended_tone}")
        st.write(f"**Platform Behavior:** {profile.platform_behavior}")


def render_campaign_plan(plan) -> None:
    st.markdown("<div class='section-category'>Agent #3: Multi-Format Campaign Strategist</div>", unsafe_allow_html=True)
    st.markdown("### 📋 Campaign Asset Deliverables Plan")

    st.markdown(
        """
        <div class='agent-thinking-card'>
            <div class='agent-thinking-title'>🧠 Agent Thinking & Rationale</div>
            <div class='agent-thinking-text'>
                Mapped creative direction into concrete ad placements (Hero, Lifestyle, Story) with targeted aspect ratios (1:1, 4:5, 9:16) and specific prompt requirements.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"Planned assets for campaign: **{plan.campaign_name}**")

    for idx, asset_plan in enumerate(plan.assets, 1):
        with st.expander(f"Asset #{idx}: `{asset_plan.asset_id}` ({asset_plan.asset_type.value})", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Placement Purpose:** {asset_plan.purpose}")
                st.write(f"**Target Audience Slice:** {asset_plan.target_audience}")
                st.write(f"**Core Message:** {asset_plan.message}")
            with c2:
                st.write(f"**Visual Direction:** {asset_plan.visual_direction}")
                st.write(f"**Aspect Ratio:** `{asset_plan.aspect_ratio}`")
                st.markdown("**Prompt Requirements:**")
                for req in asset_plan.prompt_requirements:
                    st.markdown(f"- {req}")


def render_generation_and_evaluation_loops(result) -> None:
    st.markdown("<div class='section-category'>Agent #4: Diffusion Prompt Engineer & Vision Evaluator</div>", unsafe_allow_html=True)
    st.markdown("### 🎨 Prompt Optimization & Multimodal Evaluation Loops")

    st.markdown(
        """
        <div class='agent-thinking-card'>
            <div class='agent-thinking-title'>🧠 Agent Thinking & Rationale</div>
            <div class='agent-thinking-text'>
                Engineered diffusion prompts, executed image generation, scored results via CLIP semantic cosine alignment, and revised prompts automatically upon failing quality gates.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for asset in result.assets:
        status_text = "🟢 PASSED QUALITY GATE" if asset.evaluation.passed else "🔴 EXHAUSTED ATTEMPTS"
        st.markdown("---")
        st.subheader(f"Asset `{asset.asset_id}` — {status_text}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Score", f"{asset.evaluation.overall_score:.2f}")
        m2.metric("CLIP Similarity", f"{asset.evaluation.semantic_score:.2f}")
        m3.metric("Technical Score", f"{asset.evaluation.technical_score:.2f}")
        m4.metric("Attempts Used", f"{asset.attempts_used}")

        attempts_map: dict[int, list] = {}
        for c in asset.all_candidates:
            attempts_map.setdefault(c.attempt_number, []).append(c)

        for att_num in sorted(attempts_map.keys()):
            cands = attempts_map[att_num]
            sample_prompt = cands[0].prompt if cands else asset.prompt

            with st.expander(f"🔄 Generation Attempt #{att_num} ({len(cands)} Candidates Evaluated)", expanded=(att_num == asset.attempts_used)):
                st.markdown(f"**Positive Prompt:**")
                st.code(sample_prompt.positive_prompt, language="text")

                if sample_prompt.negative_prompt:
                    st.markdown(f"**Negative Prompt:** `{sample_prompt.negative_prompt}`")

                st.caption(f"Guidance Scale: `{sample_prompt.guidance_scale}` | Steps: `{sample_prompt.num_inference_steps}` | Aspect Ratio: `{sample_prompt.aspect_ratio}`")

                cand_cols = st.columns(min(len(cands), 3))
                for idx, cand in enumerate(cands):
                    col = cand_cols[idx % len(cand_cols)]
                    with col:
                        st.image(cand.file_path, caption=cand.candidate_id, use_container_width=True)
                        if cand.evaluation:
                            st.markdown(f"Score: **{cand.evaluation.overall_score:.2f}** (Passed: `{cand.evaluation.passed}`)")
                            st.caption(f"CLIP: `{cand.evaluation.semantic_score:.2f}` | Tech: `{cand.evaluation.technical_score:.2f}`")
                            if cand.evaluation.feedback:
                                st.write("**Feedback:** " + "; ".join(cand.evaluation.feedback))

                best_cand = max(cands, key=lambda x: x.evaluation.overall_score if x.evaluation else 0.0)
                if best_cand.evaluation and not best_cand.evaluation.passed:
                    st.warning(
                        f"⚠️ Attempt #{att_num} failed quality gate (score: {best_cand.evaluation.overall_score:.2f}). "
                        f"Optimizer feedback for next attempt: {'; '.join(best_cand.evaluation.feedback or ['Improve alignment'])}"
                    )
                else:
                    st.success(f"✅ Attempt #{att_num} met quality gate threshold! Selected for delivery.")


def render_copywriting(copy) -> None:
    st.markdown("<div class='section-category'>Agent #5: Creative Copywriter</div>", unsafe_allow_html=True)
    st.markdown("### ✍️ Social Copywriting & Deliverables")

    st.markdown(
        """
        <div class='agent-thinking-card'>
            <div class='agent-thinking-title'>🧠 Agent Thinking & Rationale</div>
            <div class='agent-thinking-text'>
                Drafted high-converting headlines, body captions, calls-to-action, and targeted hashtags per visual asset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for asset_copy in copy.assets:
        with st.container():
            st.markdown(f"#### Asset Copy: `{asset_copy.asset_id}`")
            st.markdown(f"**Headline:** `{asset_copy.headline}`")
            st.markdown(f"**Caption:**\n{asset_copy.caption}")
            st.markdown(f"**Call To Action:** `{asset_copy.cta}`")
            tags_html = "".join(f"<span class='tag-pill'>#{h.lstrip('#')}</span>" for h in asset_copy.hashtags)
            st.markdown(tags_html, unsafe_allow_html=True)
            st.markdown("---")


def render_logs_and_thinking(log_records: list[str]) -> None:
    st.markdown("<div class='section-category'>Execution Log Stream</div>", unsafe_allow_html=True)
    st.markdown("### 📜 Real-Time Agent Execution Logs")

    search = st.text_input("Filter execution logs", "")
    filtered = [r for r in log_records if search.lower() in r.lower()] if search else log_records

    st.code("\n".join(filtered) if filtered else "No matching log records found.", language="text")


def render_package_export(result, copy, campaign_dir) -> None:
    st.markdown("<div class='section-category'>Export Deliverables</div>", unsafe_allow_html=True)
    st.markdown("### 📦 Campaign Package & JSON Exports")
    st.write(f"Campaign Package Path: `{campaign_dir}`")
    st.markdown("---")

    t1, t2, t3 = st.tabs(["Creative Brief JSON", "Campaign Plan JSON", "Campaign Copy JSON"])
    with t1:
        st.json(result.brief.model_dump())
    with t2:
        st.json(result.plan.model_dump())
    with t3:
        st.json(copy.model_dump())


def main() -> None:
    st.set_page_config(page_title="CampaignStudio", layout="wide")
    inject_custom_css()

    st.markdown("<h1 class='studio-title-main'>CampaignStudio</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='studio-subtitle'>Autonomous Multi-Agent Creative & Media Studio — Fully Local Execution</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Campaign Inputs")
        campaign_brief = st.text_area(
            "Campaign brief",
            placeholder="Launch a campaign for an eco-friendly running shoe...",
            height=140,
        )
        target_audience = st.text_input("Target audience (optional)", "")
        brand_style = st.text_input("Brand style / personality (optional)", "")
        number_of_assets = st.slider("Number of assets", min_value=1, max_value=8, value=3)

        st.header("Execution Settings")
        ollama_model = st.text_input("Ollama model", settings.ollama_model)
        quality_threshold = st.slider(
            "Quality threshold", 0.0, 1.0, settings.quality_threshold, 0.05
        )

        run_clicked = st.button("Generate Campaign", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Fill in a campaign brief and click **Generate campaign** to start.")
        return

    if not campaign_brief.strip():
        st.error("Campaign brief cannot be empty.")
        return

    if not check_ollama_available():
        st.error(
            f"Ollama is not reachable at {settings.ollama_base_url}. Start it with "
            f"`ollama serve` and pull the model with `ollama pull {ollama_model}`."
        )
        return

    apply_ui_overrides(ollama_model, quality_threshold)
    full_brief_text = build_full_brief_text(campaign_brief, target_audience, brand_style)

    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        with st.status("🚀 Running CampaignStudio Pipeline...", expanded=True) as status:
            st.write("🧠 Creative Director: Developing strategy brief...")
            st.write("👥 Audience Analyst: Profiling target audience...")
            st.write("📋 Campaign Strategist: Structuring asset deliverables...")
            st.write("🎨 Prompt Optimizer & Generator: Running local diffusion & CLIP evaluation loops...")
            st.write("✍️ Copywriter: Writing social headlines & captions...")
            
            result, copy, campaign_dir = run_full_campaign(
                full_brief_text, number_of_assets=number_of_assets
            )
            status.update(label="✅ Campaign Generated Successfully!", state="complete", expanded=False)
    except CreativeDirectorError as exc:
        st.error(str(exc))
        return
    finally:
        root_logger.removeHandler(handler)

    (
        tab_demo,
        tab_summary,
        tab_brief,
        tab_audience,
        tab_plan,
        tab_gen,
        tab_copy,
        tab_logs,
        tab_export,
    ) = st.tabs(
        [
            "🎬 Demo Showcase",
            "📊 Executive Summary",
            "🎯 Creative Brief",
            "👥 Audience Insights",
            "📋 Campaign Plan",
            "🎨 Generation & Loops",
            "✍️ Copywriting",
            "📜 Execution Logs",
            "📦 Export Package",
        ]
    )

    with tab_demo:
        render_demo_showcase(result, copy)

    with tab_summary:
        render_executive_summary(result, copy, campaign_dir)

    with tab_brief:
        render_creative_brief(result.brief)

    with tab_audience:
        render_audience_insights(result.brief)

    with tab_plan:
        render_campaign_plan(result.plan)

    with tab_gen:
        render_generation_and_evaluation_loops(result)

    with tab_copy:
        render_copywriting(copy)

    with tab_logs:
        render_logs_and_thinking(handler.records)

    with tab_export:
        render_package_export(result, copy, campaign_dir)


if __name__ == "__main__":
    main()
