"""
Streamlit UI for the AI Creative Director.

Deliberately thin: this file's only job is collecting inputs, calling
services.campaign_service.run_full_campaign(), and rendering the results.
All real logic (agents, workflow, evaluation) lives outside this file --
project brief section 20 is explicit that "the AI workflow is the
project," not the UI.

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from config import settings
from models.exceptions import CreativeDirectorError
from models.llm import check_ollama_available
from services.campaign_service import run_full_campaign


class StreamlitLogHandler(logging.Handler):
    """Collects formatted log records into a list for display after the
    (synchronous) campaign run completes. Extracted as its own class so it
    can be unit-tested without a running Streamlit app."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


def apply_ui_overrides(image_backend: str, ollama_model: str, quality_threshold: float) -> None:
    """Mutate the shared settings singleton for this run.

    This is a single-user local tool (project brief section 31: don't
    overengineer), so an in-place override on the settings singleton is
    enough -- no per-session config store is needed. Extracted as its own
    function so it's unit-testable independent of Streamlit widgets.
    """
    settings.image_backend = image_backend
    settings.ollama_model = ollama_model
    settings.quality_threshold = quality_threshold


def build_full_brief_text(campaign_brief: str, target_audience: str, brand_style: str) -> str:
    """Fold optional UI-only fields (target audience, brand style) into the
    single brief string the Creative Director agent consumes. Kept as a
    pure function so it's testable without Streamlit."""
    text = campaign_brief
    if target_audience:
        text += f"\n\nTarget audience: {target_audience}"
    if brand_style:
        text += f"\nBrand style/personality: {brand_style}"
    return text


def render_asset(asset, asset_copy) -> None:
    """Render one generated asset: image, prompt, scores, and its copy."""
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

    if len(asset.all_candidates) > 1:
        with st.expander(f"All {len(asset.all_candidates)} candidates for {asset.asset_id}"):
            for candidate in asset.all_candidates:
                if candidate.evaluation is not None:
                    st.write(
                        f"`{candidate.candidate_id}` — attempt {candidate.attempt_number} — "
                        f"score: {candidate.evaluation.overall_score:.2f}"
                    )
                else:
                    st.write(f"`{candidate.candidate_id}` — no evaluation available")
                st.image(candidate.file_path, width=200)


def main() -> None:
    st.set_page_config(page_title="AI Creative Director", layout="wide")
    st.title("AI Creative Director")
    st.caption(
        "A local, multi-agent creative team: campaign strategy -> prompt "
        "optimization -> local diffusion generation -> automated evaluation "
        "-> regeneration -> copywriting. Everything below runs on your machine."
    )

    with st.sidebar:
        st.header("Campaign inputs")
        campaign_brief = st.text_area(
            "Campaign brief",
            placeholder="Create a marketing campaign for an eco-friendly running shoe...",
            height=140,
        )
        target_audience = st.text_input("Target audience (optional)", "")
        brand_style = st.text_input("Brand style / personality (optional)", "")
        number_of_assets = st.slider("Number of assets", min_value=1, max_value=8, value=3)

        st.header("Generation settings")
        image_backend = st.selectbox("Image backend", ["diffusers", "comfyui"], index=0)
        ollama_model = st.text_input("Ollama model", settings.ollama_model)
        quality_threshold = st.slider(
            "Quality threshold", 0.0, 1.0, settings.quality_threshold, 0.05
        )

        run_clicked = st.button("Generate campaign", type="primary")

    if not run_clicked:
        st.info("Fill in a campaign brief and click **Generate campaign** to start.")
        return

    if not campaign_brief.strip():
        st.error("Campaign brief cannot be empty.")
        return

    if not check_ollama_available():
        st.error(
            f"Ollama is not reachable at {settings.ollama_base_url}. Start it with "
            f"`ollama serve` and pull the configured model with `ollama pull {ollama_model}`."
        )
        return

    apply_ui_overrides(image_backend, ollama_model, quality_threshold)
    full_brief_text = build_full_brief_text(campaign_brief, target_audience, brand_style)

    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        with st.spinner(
            "Running the creative team... this can take a while, especially on "
            "first image generation (model download/load)."
        ):
            result, copy, campaign_dir = run_full_campaign(
                full_brief_text, number_of_assets=number_of_assets
            )
    except CreativeDirectorError as exc:
        st.error(str(exc))
        return
    finally:
        root_logger.removeHandler(handler)

    st.success(f"Campaign package written to `{campaign_dir}`")

    st.header("1. Creative brief")
    st.json(result.brief.model_dump())

    st.header("2. Agent workflow log")
    with st.expander("Show full log", expanded=False):
        st.code("\n".join(handler.records))

    st.header("3. Generated assets")
    copy_by_asset = {c.asset_id: c for c in copy.assets}
    for asset in result.assets:
        render_asset(asset, copy_by_asset.get(asset.asset_id))

    st.header("4. Full campaign copy (JSON)")
    st.json(copy.model_dump())


if __name__ == "__main__":
    main()
