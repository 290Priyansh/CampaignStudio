from __future__ import annotations

import pytest

from models.exceptions import OllamaUnavailableError
from schemas import (
    AssetPlan,
    AssetType,
    CampaignPlan,
    Candidate,
    CreativeBrief,
    EvaluationResult,
    GeneratedAsset,
    ImagePrompt,
)
from workflows import campaign_workflow


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
        number_of_assets=2,
        recommended_aspect_ratios=["4:5"],
        content_ideas=["hero shot"],
        image_generation_strategy="product-in-context",
    )


def _plan() -> CampaignPlan:
    return CampaignPlan(
        campaign_name="EcoStride Launch",
        assets=[
            AssetPlan(
                asset_id="hero", asset_type=AssetType.HERO_IMAGE, purpose="Anchor visual",
                target_audience="Students", message="msg", visual_direction="dir",
                aspect_ratio="4:5", prompt_requirements=["req"],
            ),
            AssetPlan(
                asset_id="lifestyle", asset_type=AssetType.LIFESTYLE_IMAGE, purpose="Context",
                target_audience="Students", message="msg2", visual_direction="dir2",
                aspect_ratio="4:5", prompt_requirements=["req2"],
            ),
        ],
    )


def _generated_asset(asset_id: str) -> GeneratedAsset:
    prompt = ImagePrompt(
        positive_prompt="p", negative_prompt="n", aspect_ratio="4:5",
        style="s", lighting="l", composition="c", subject="subj",
    )
    evaluation = EvaluationResult(semantic_score=0.8, technical_score=0.9, overall_score=0.85, passed=True)
    candidate = Candidate(candidate_id=f"{asset_id}_c1", file_path=f"/tmp/{asset_id}.png", prompt=prompt, attempt_number=1, evaluation=evaluation)
    return GeneratedAsset(
        asset_id=asset_id, file_path=f"/tmp/{asset_id}.png", prompt=prompt,
        evaluation=evaluation, attempts_used=1, all_candidates=[candidate],
    )


def test_run_campaign_raises_if_ollama_unavailable(monkeypatch):
    monkeypatch.setattr(campaign_workflow, "check_ollama_available", lambda: False)
    with pytest.raises(OllamaUnavailableError):
        campaign_workflow.run_campaign("some brief")


def test_run_campaign_chains_agents_and_generates_all_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign_workflow, "check_ollama_available", lambda: True)

    brief = _brief()
    plan = _plan()

    monkeypatch.setattr(
        campaign_workflow.creative_director, "create_brief",
        lambda text, number_of_assets: brief,
    )

    audience_calls = []

    def fake_analyze(b):
        audience_calls.append(b)
        from schemas import AudienceProfile
        return AudienceProfile(
            demographic_summary="x", motivations=["m"], pain_points=["p"],
            desired_emotional_response="e", platform_behavior="b", recommended_tone="t",
        )

    monkeypatch.setattr(campaign_workflow.audience_analyst, "analyze_audience", fake_analyze)
    monkeypatch.setattr(campaign_workflow.campaign_strategist, "build_campaign_plan", lambda b: plan)
    monkeypatch.setattr(campaign_workflow, "get_image_generator", lambda: object())

    generate_calls = []

    def fake_generate_asset(brief_arg, asset_plan, image_generator, output_dir):
        generate_calls.append(asset_plan.asset_id)
        return _generated_asset(asset_plan.asset_id)

    monkeypatch.setattr(campaign_workflow, "generate_asset", fake_generate_asset)

    result = campaign_workflow.run_campaign("eco shoe campaign", number_of_assets=2, output_dir=tmp_path)

    assert result.brief.campaign_name == "EcoStride Launch"
    assert result.brief.audience_profile is not None
    assert len(audience_calls) == 1
    assert generate_calls == ["hero", "lifestyle"]
    assert len(result.assets) == 2
    assert {a.asset_id for a in result.assets} == {"hero", "lifestyle"}


def test_slugify_produces_filesystem_safe_names():
    assert campaign_workflow._slugify("EcoStride Launch!") == "ecostride_launch"
    assert campaign_workflow._slugify("  Multi   Space  ") == "multi_space"
