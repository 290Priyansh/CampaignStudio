from __future__ import annotations

import pytest

from agents import campaign_strategist
from models.exceptions import CrewExecutionError
from schemas import AssetPlan, AssetType, AudienceProfile, CampaignPlan, CreativeBrief


def _brief(with_audience: bool = True) -> CreativeBrief:
    brief = CreativeBrief(
        campaign_name="EcoStride Launch",
        campaign_objective="Drive awareness for a new eco-friendly running shoe.",
        target_audience="College students, 18-24",
        key_message="Move fast, tread light.",
        brand_personality=["energetic"],
        visual_style="bright editorial photography",
        color_direction=["forest green"],
        typography_direction="bold sans-serif",
        composition_guidelines="rule of thirds",
        photography_style="natural light",
        number_of_assets=3,
        recommended_aspect_ratios=["4:5", "9:16"],
        content_ideas=["hero shot", "campus lifestyle"],
        image_generation_strategy="product-in-context",
    )
    if with_audience:
        brief.audience_profile = AudienceProfile(
            demographic_summary="College students balancing budget and identity",
            motivations=["self-expression"],
            pain_points=["greenwashing skepticism"],
            desired_emotional_response="aspirational confidence",
            platform_behavior="fast scroll",
            recommended_tone="playful but credible",
        )
    return brief


def _plan() -> CampaignPlan:
    return CampaignPlan(
        campaign_name="EcoStride Launch",
        assets=[
            AssetPlan(
                asset_id="hero",
                asset_type=AssetType.HERO_IMAGE,
                purpose="Anchor visual",
                target_audience="College students",
                message="Statement of values",
                visual_direction="dramatic side-lit shot",
                aspect_ratio="4:5",
                prompt_requirements=["product in focus"],
            )
        ],
    )


def test_build_campaign_plan_returns_validated_plan(fake_crewai):
    expected = _plan()
    fake_crewai._pending_result = expected

    result = campaign_strategist.build_campaign_plan(_brief())

    assert result is expected


def test_build_campaign_plan_task_includes_audience_context_when_present(fake_crewai, monkeypatch):
    fake_crewai._pending_result = _plan()
    captured = {}
    real_init = fake_crewai.Task.__init__

    def spying_init(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(fake_crewai.Task, "__init__", spying_init)

    campaign_strategist.build_campaign_plan(_brief(with_audience=True))

    assert "aspirational confidence" in captured["description"]
    assert "exactly 3 assets" in captured["description"]
    assert captured["output_pydantic"] is CampaignPlan


def test_build_campaign_plan_task_omits_audience_context_when_absent(fake_crewai, monkeypatch):
    fake_crewai._pending_result = _plan()
    captured = {}
    real_init = fake_crewai.Task.__init__

    def spying_init(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(fake_crewai.Task, "__init__", spying_init)

    campaign_strategist.build_campaign_plan(_brief(with_audience=False))

    assert "Audience insight" not in captured["description"]


def test_build_campaign_plan_raises_on_crew_failure(fake_crewai):
    fake_crewai._raise_on_kickoff = "model unavailable"
    with pytest.raises(CrewExecutionError):
        campaign_strategist.build_campaign_plan(_brief())


def test_build_campaign_plan_raises_when_no_structured_output(fake_crewai):
    fake_crewai._pending_result = None
    with pytest.raises(CrewExecutionError):
        campaign_strategist.build_campaign_plan(_brief())
