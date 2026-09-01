from __future__ import annotations

import pytest

from agents import audience_analyst
from models.exceptions import CrewExecutionError
from schemas import AudienceProfile, CreativeBrief


def _brief() -> CreativeBrief:
    return CreativeBrief(
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
        recommended_aspect_ratios=["4:5"],
        content_ideas=["hero shot"],
        image_generation_strategy="product-in-context",
    )


def _profile() -> AudienceProfile:
    return AudienceProfile(
        demographic_summary="College students balancing budget and identity",
        motivations=["self-expression", "value for money"],
        pain_points=["greenwashing skepticism"],
        desired_emotional_response="aspirational confidence",
        platform_behavior="fast scroll, stops for bold color and motion",
        recommended_tone="playful but credible",
    )


def test_analyze_audience_returns_validated_profile(fake_crewai):
    expected = _profile()
    fake_crewai._pending_result = expected

    result = audience_analyst.analyze_audience(_brief())

    assert result is expected


def test_analyze_audience_task_includes_brief_context(fake_crewai, monkeypatch):
    fake_crewai._pending_result = _profile()
    captured = {}
    real_init = fake_crewai.Task.__init__

    def spying_init(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(fake_crewai.Task, "__init__", spying_init)

    audience_analyst.analyze_audience(_brief())

    assert "EcoStride Launch" in captured["description"]
    assert "College students, 18-24" in captured["description"]
    assert captured["output_pydantic"] is AudienceProfile


def test_analyze_audience_raises_on_crew_failure(fake_crewai):
    fake_crewai._raise_on_kickoff = "timeout"
    with pytest.raises(CrewExecutionError):
        audience_analyst.analyze_audience(_brief())


def test_analyze_audience_raises_when_no_structured_output(fake_crewai):
    fake_crewai._pending_result = None
    with pytest.raises(CrewExecutionError):
        audience_analyst.analyze_audience(_brief())
