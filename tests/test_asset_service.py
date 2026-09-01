"""
Tests for services/asset_service.py -- verifies the on-disk campaign
package layout matches project brief section 19 exactly.
"""

from __future__ import annotations

import json

from schemas import (
    AssetPlan,
    AssetType,
    CampaignCopy,
    AssetCopy,
    CampaignPlan,
    CreativeBrief,
    EvaluationResult,
    GeneratedAsset,
    ImagePrompt,
)
from services.asset_service import package_campaign
from workflows.campaign_workflow import CampaignResult


def _brief() -> CreativeBrief:
    return CreativeBrief(
        campaign_name="EcoStride Launch!",
        campaign_objective="Drive awareness",
        target_audience="College students",
        key_message="Move fast, tread light.",
        brand_personality=["energetic"],
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


def _plan() -> CampaignPlan:
    return CampaignPlan(
        campaign_name="EcoStride Launch!",
        assets=[
            AssetPlan(
                asset_id="hero", asset_type=AssetType.HERO_IMAGE, purpose="Anchor visual",
                target_audience="College students", message="Statement of values",
                visual_direction="dramatic side-lit shot", aspect_ratio="4:5",
                prompt_requirements=["product in focus"],
            )
        ],
    )


def _generated_asset(tmp_path) -> GeneratedAsset:
    source_image = tmp_path / "raw_candidate.png"
    source_image.write_bytes(b"FAKEPNGDATA")

    prompt = ImagePrompt(
        positive_prompt="a running shoe on a forest trail, golden hour",
        negative_prompt="blurry", aspect_ratio="4:5", style="editorial",
        lighting="golden hour", composition="rule of thirds", subject="running shoe",
    )
    evaluation = EvaluationResult(semantic_score=0.9, technical_score=1.0, overall_score=0.92, passed=True)
    return GeneratedAsset(
        asset_id="hero", file_path=str(source_image), prompt=prompt,
        evaluation=evaluation, attempts_used=1,
    )


def test_package_campaign_creates_expected_directory_layout(tmp_path):
    result = CampaignResult(brief=_brief(), plan=_plan(), assets=[_generated_asset(tmp_path)])

    campaign_dir = package_campaign(result, output_dir=tmp_path / "outputs")

    assert campaign_dir.name == "ecostride_launch"  # slugified, matches _slugify behavior
    assert (campaign_dir / "campaign_brief.json").exists()
    assert (campaign_dir / "strategy.json").exists()
    assert (campaign_dir / "images" / "hero.png").exists()
    assert (campaign_dir / "prompts" / "hero.txt").exists()
    assert (campaign_dir / "evaluations" / "hero.json").exists()
    assert not (campaign_dir / "campaign_copy.json").exists()  # no copy passed


def test_package_campaign_writes_valid_json_content(tmp_path):
    result = CampaignResult(brief=_brief(), plan=_plan(), assets=[_generated_asset(tmp_path)])
    campaign_dir = package_campaign(result, output_dir=tmp_path / "outputs")

    brief_data = json.loads((campaign_dir / "campaign_brief.json").read_text())
    assert brief_data["campaign_name"] == "EcoStride Launch!"

    eval_data = json.loads((campaign_dir / "evaluations" / "hero.json").read_text())
    assert eval_data["overall_score"] == 0.92
    assert eval_data["passed"] is True


def test_package_campaign_prompt_file_contains_positive_and_negative(tmp_path):
    result = CampaignResult(brief=_brief(), plan=_plan(), assets=[_generated_asset(tmp_path)])
    campaign_dir = package_campaign(result, output_dir=tmp_path / "outputs")

    prompt_text = (campaign_dir / "prompts" / "hero.txt").read_text()
    assert "a running shoe on a forest trail, golden hour" in prompt_text
    assert "blurry" in prompt_text


def test_package_campaign_writes_copy_when_provided(tmp_path):
    result = CampaignResult(brief=_brief(), plan=_plan(), assets=[_generated_asset(tmp_path)])
    copy = CampaignCopy(
        campaign_name="EcoStride Launch!",
        assets=[
            AssetCopy(
                asset_id="hero", headline="Tread Light", caption="Move fast, tread light.",
                cta="Shop now.", hashtags=["#eco", "#running", "#shoes", "#launch", "#green"],
            )
        ],
    )

    campaign_dir = package_campaign(result, output_dir=tmp_path / "outputs", copy=copy)

    assert (campaign_dir / "campaign_copy.json").exists()
    copy_data = json.loads((campaign_dir / "campaign_copy.json").read_text())
    assert copy_data["assets"][0]["headline"] == "Tread Light"


def test_package_campaign_survives_missing_source_image(tmp_path, monkeypatch):
    """If the source candidate file has vanished, packaging should log and
    continue rather than crash the whole campaign export."""
    result = CampaignResult(brief=_brief(), plan=_plan(), assets=[_generated_asset(tmp_path)])
    import os
    os.remove(result.assets[0].file_path)

    campaign_dir = package_campaign(result, output_dir=tmp_path / "outputs")

    assert not (campaign_dir / "images" / "hero.png").exists()
    assert (campaign_dir / "campaign_brief.json").exists()  # rest of packaging still succeeded
