"""
Tests for agents/copywriter.py, using the shared `fake_crewai` fixture
(tests/conftest.py) already established by the other three agent test
files in this project.
"""

from __future__ import annotations

import pytest

from agents import copywriter
from models.exceptions import CrewExecutionError
from schemas import (
    AssetCopy,
    AssetPlan,
    AssetType,
    CampaignCopy,
    CreativeBrief,
    EvaluationResult,
    GeneratedAsset,
    ImagePrompt,
)


def _brief() -> CreativeBrief:
    return CreativeBrief(
        campaign_name="EcoStride Launch",
        campaign_objective="Drive awareness",
        target_audience="College students",
        key_message="Move fast, tread light.",
        brand_personality=["energetic", "modern"],
        visual_style="bright editorial",
        color_direction=["forest green"],
        typography_direction="bold sans-serif",
        composition_guidelines="rule of thirds",
        photography_style="natural light",
        number_of_assets=1,
        recommended_aspect_ratios=["4:5"],
        content_ideas=["hero shot"],
        image_generation_strategy="product-in-context",
    )


def _asset_plan() -> AssetPlan:
    return AssetPlan(
        asset_id="hero",
        asset_type=AssetType.HERO_IMAGE,
        purpose="Anchor visual",
        target_audience="College students",
        message="The shoe as a statement of values",
        visual_direction="dramatic side-lit shot",
        aspect_ratio="4:5",
        prompt_requirements=["product in focus"],
    )


def _generated_asset() -> GeneratedAsset:
    prompt = ImagePrompt(
        positive_prompt="a running shoe on a forest trail, golden hour",
        negative_prompt="blurry", aspect_ratio="4:5", style="editorial",
        lighting="golden hour", composition="rule of thirds", subject="running shoe",
    )
    evaluation = EvaluationResult(semantic_score=0.9, technical_score=1.0, overall_score=0.92, passed=True)
    return GeneratedAsset(
        asset_id="hero", file_path="outputs/x/images/hero.png", prompt=prompt,
        evaluation=evaluation, attempts_used=1,
    )


def _fake_copy() -> CampaignCopy:
    return CampaignCopy(
        campaign_name="EcoStride Launch",
        assets=[
            AssetCopy(
                asset_id="hero", headline="Tread Light",
                caption="Every step forward, lighter on the planet.",
                cta="Shop the launch.",
                hashtags=["#ecostride", "#sustainable", "#running", "#collegelife", "#launch"],
            )
        ],
    )


def test_write_campaign_copy_returns_validated_result(fake_crewai):
    fake_crewai._pending_result = _fake_copy()

    result = copywriter.write_campaign_copy(_brief(), [_asset_plan()], [_generated_asset()])

    assert result is fake_crewai._pending_result
    assert isinstance(result, CampaignCopy)


def test_write_campaign_copy_task_grounds_in_actual_prompt(fake_crewai, monkeypatch):
    fake_crewai._pending_result = _fake_copy()
    captured = {}

    real_task_init = fake_crewai.Task.__init__

    def spying_init(self, **kwargs):
        captured.update(kwargs)
        real_task_init(self, **kwargs)

    monkeypatch.setattr(fake_crewai.Task, "__init__", spying_init)

    copywriter.write_campaign_copy(_brief(), [_asset_plan()], [_generated_asset()])

    description = captured["description"]
    assert "a running shoe on a forest trail, golden hour" in description
    assert "hero" in description
    assert "0.92" in description
    assert captured["output_pydantic"] is CampaignCopy


def test_write_campaign_copy_raises_on_crew_failure(fake_crewai):
    fake_crewai._raise_on_kickoff = "Ollama connection refused"

    with pytest.raises(CrewExecutionError):
        copywriter.write_campaign_copy(_brief(), [_asset_plan()], [_generated_asset()])


def test_write_campaign_copy_raises_when_no_structured_output(fake_crewai):
    fake_crewai._pending_result = None

    with pytest.raises(CrewExecutionError):
        copywriter.write_campaign_copy(_brief(), [_asset_plan()], [_generated_asset()])
