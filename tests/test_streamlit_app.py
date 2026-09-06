"""
Tests for ui/streamlit_app.py.

Pure logic (StreamlitLogHandler, apply_ui_overrides, build_full_brief_text)
is unit-tested directly with no Streamlit runtime involved. The full app
is additionally smoke-tested with Streamlit's own `AppTest` harness
(streamlit.testing.v1), which actually executes the script in a simulated
session -- this catches import errors, widget-wiring mistakes, and crashes
on the "not yet run" path without needing a real browser or Ollama server.
"""

from __future__ import annotations

import logging

from config.settings import Settings
from ui.streamlit_app import apply_ui_overrides, build_full_brief_text, StreamlitLogHandler


def test_streamlit_log_handler_collects_formatted_records():
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger = logging.getLogger("TestComponent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("hello %s", "world")

    assert handler.records == ["[TestComponent] hello world"]
    logger.removeHandler(handler)


def test_apply_ui_overrides_mutates_settings_singleton(monkeypatch):
    import ui.streamlit_app as app_module

    fake_settings = Settings(_env_file=None)
    monkeypatch.setattr(app_module, "settings", fake_settings)

    apply_ui_overrides("gemma3:4b", 0.6)

    assert fake_settings.ollama_model == "gemma3:4b"
    assert fake_settings.quality_threshold == 0.6


def test_build_full_brief_text_appends_optional_fields():
    text = build_full_brief_text("Launch an eco shoe", "College students", "energetic, modern")
    assert text.startswith("Launch an eco shoe")
    assert "Target audience: College students" in text
    assert "Brand style/personality: energetic, modern" in text


def test_build_full_brief_text_omits_empty_optional_fields():
    text = build_full_brief_text("Launch an eco shoe", "", "")
    assert text == "Launch an eco shoe"


def test_app_smoke_renders_initial_state_without_running_campaign():
    """Full AppTest run: the script should execute cleanly and show the
    'fill in a brief' prompt before the Generate button is clicked -- no
    network, Ollama, or GPU access should occur on this path."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("ui/streamlit_app.py")
    at.run()

    assert not at.exception
    assert any("Fill in a campaign brief" in info.value for info in at.info)


def test_app_smoke_errors_on_empty_brief_when_run_clicked():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("ui/streamlit_app.py")
    at.run()
    at.button[0].click().run()

    assert not at.exception
    assert any("cannot be empty" in err.value for err in at.error)


def test_render_functions_execute_without_error(tmp_path):
    from schemas import (
        AssetCopy,
        AssetPlan,
        AssetType,
        AudienceProfile,
        CampaignCopy,
        CampaignPlan,
        Candidate,
        CreativeBrief,
        EvaluationResult,
        GeneratedAsset,
        ImagePrompt,
    )
    from ui.streamlit_app import (
        render_audience_insights,
        render_campaign_plan,
        render_copywriting,
        render_creative_brief,
        render_demo_showcase,
        render_executive_summary,
        render_generation_and_evaluation_loops,
        render_logs_and_thinking,
        render_package_export,
    )
    from workflows.campaign_workflow import CampaignResult

    brief = CreativeBrief(
        campaign_name="Eco Runner",
        campaign_objective="Launch eco shoe",
        target_audience="Runners",
        key_message="Sustainable speed",
        brand_personality=["energetic", "sustainable"],
        visual_style="Editorial",
        color_direction=["green"],
        typography_direction="Clean sans-serif",
        composition_guidelines="Rule of thirds",
        photography_style="Natural light",
        number_of_assets=1,
        recommended_aspect_ratios=["1:1"],
        content_ideas=["Shoe in nature"],
        image_generation_strategy="Focus on materials",
        audience_profile=AudienceProfile(
            demographic_summary="Eco runners 20-35",
            motivations=["Sustainability"],
            pain_points=["Plastic waste"],
            desired_emotional_response="Inspiration",
            platform_behavior="High engagement",
            recommended_tone="Empowering",
        ),
    )

    plan = CampaignPlan(
        campaign_name="Eco Runner",
        assets=[
            AssetPlan(
                asset_id="hero_01",
                asset_type=AssetType.HERO_IMAGE,
                purpose="Lead ad",
                target_audience="Runners",
                message="Run green",
                visual_direction="Bright forest trail",
                aspect_ratio="1:1",
                prompt_requirements=["green shoe", "nature trail"],
            )
        ],
    )

    from PIL import Image

    dummy_image = tmp_path / "hero.png"
    Image.new("RGB", (100, 100), color="red").save(dummy_image)

    prompt = ImagePrompt(
        positive_prompt="Eco shoe on forest trail",
        negative_prompt="blurry",
        aspect_ratio="1:1",
        subject="Eco running shoe",
        style="Editorial photograph",
        lighting="Natural daylight",
        composition="Centered hero shot",
    )

    evaluation = EvaluationResult(
        semantic_score=0.85,
        technical_score=0.9,
        overall_score=0.87,
        passed=True,
    )

    generated_asset = GeneratedAsset(
        asset_id="hero_01",
        file_path=str(dummy_image),
        prompt=prompt,
        evaluation=evaluation,
        attempts_used=1,
        all_candidates=[
            Candidate(
                candidate_id="hero_01_a1_c1",
                file_path=str(dummy_image),
                prompt=prompt,
                attempt_number=1,
                evaluation=evaluation,
            )
        ],
    )

    campaign_result = CampaignResult(brief=brief, plan=plan, assets=[generated_asset])
    campaign_copy = CampaignCopy(
        campaign_name="Eco Runner",
        assets=[
            AssetCopy(
                asset_id="hero_01",
                headline="Step Into Tomorrow",
                caption="Meet the eco shoe.",
                cta="Shop Now",
                hashtags=["#EcoRunner", "#Sustainable", "#EcoShoe", "#RunGreen", "#ZeroWaste"],
            )
        ],
    )

    # Calling render functions directly verifies structure & formatting without needing live Streamlit server
    render_executive_summary(campaign_result, campaign_copy, tmp_path / "pkg")
    render_creative_brief(brief)
    render_audience_insights(brief)
    render_campaign_plan(plan)
    render_generation_and_evaluation_loops(campaign_result)
    render_copywriting(campaign_copy)
    render_logs_and_thinking(["[Workflow] Test log record"])
    render_package_export(campaign_result, campaign_copy, tmp_path / "pkg")

