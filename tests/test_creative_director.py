from __future__ import annotations

import pytest

from agents import creative_director
from models.exceptions import CrewExecutionError
from schemas import CreativeBrief


def _valid_brief() -> CreativeBrief:
    return CreativeBrief(
        campaign_name="EcoStride Launch",
        campaign_objective="Drive awareness for a new eco-friendly running shoe.",
        target_audience="College students, 18-24",
        key_message="Move fast, tread light.",
        brand_personality=["energetic", "modern"],
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


def test_create_brief_returns_validated_brief(fake_crewai):
    expected = _valid_brief()
    fake_crewai._pending_result = expected

    result = creative_director.create_brief(
        "Create a campaign for an eco-friendly running shoe.", number_of_assets=3
    )

    assert result is expected


def test_create_brief_task_description_includes_input_and_count(fake_crewai, monkeypatch):
    fake_crewai._pending_result = _valid_brief()
    captured = {}

    real_task_init = fake_crewai.Task.__init__

    def spying_init(self, **kwargs):
        captured.update(kwargs)
        real_task_init(self, **kwargs)

    monkeypatch.setattr(fake_crewai.Task, "__init__", spying_init)

    creative_director.create_brief("eco-friendly running shoe campaign", number_of_assets=7)

    assert "eco-friendly running shoe campaign" in captured["description"]
    assert "7-asset" in captured["description"]
    assert captured["output_pydantic"] is CreativeBrief


def test_create_brief_raises_on_crew_failure(fake_crewai):
    fake_crewai._raise_on_kickoff = "Ollama connection refused"
    with pytest.raises(CrewExecutionError):
        creative_director.create_brief("some brief")


def test_create_brief_raises_when_no_structured_output(fake_crewai):
    fake_crewai._pending_result = None
    with pytest.raises(CrewExecutionError):
        creative_director.create_brief("some brief")
