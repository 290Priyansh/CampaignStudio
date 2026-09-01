"""
Tests for services/campaign_service.py. Each of the three collaborators
(run_campaign, write_campaign_copy, package_campaign) is mocked at the
module level this file imports it into -- this test's only job is proving
they're called in the right order with the right arguments, not
re-testing their own internals (covered by their dedicated test files).
"""

from __future__ import annotations

from schemas import (
    AssetCopy,
    AssetPlan,
    AssetType,
    CampaignCopy,
    CampaignPlan,
    CreativeBrief,
)
from services import campaign_service
from workflows.campaign_workflow import CampaignResult


def _brief() -> CreativeBrief:
    return CreativeBrief(
        campaign_name="EcoStride Launch",
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
        campaign_name="EcoStride Launch",
        assets=[
            AssetPlan(
                asset_id="hero", asset_type=AssetType.HERO_IMAGE, purpose="Anchor visual",
                target_audience="College students", message="Statement of values",
                visual_direction="dramatic side-lit shot", aspect_ratio="4:5",
                prompt_requirements=["product in focus"],
            )
        ],
    )


def test_run_full_campaign_chains_dependencies_in_order(monkeypatch, tmp_path):
    call_order = []

    fake_result = CampaignResult(brief=_brief(), plan=_plan(), assets=[])

    def fake_run_campaign(text, number_of_assets, output_dir):
        call_order.append("run_campaign")
        assert text == "a campaign brief"
        assert number_of_assets == 3
        return fake_result

    fake_copy = CampaignCopy(
        campaign_name="EcoStride Launch",
        assets=[
            AssetCopy(
                asset_id="hero", headline="Tread Light", caption="Move fast, tread light.",
                cta="Shop now.", hashtags=["#eco", "#running", "#shoes", "#launch", "#green"],
            )
        ],
    )

    def fake_write_campaign_copy(brief, asset_plans, generated_assets):
        call_order.append("write_campaign_copy")
        assert brief is fake_result.brief
        assert asset_plans is fake_result.plan.assets
        assert generated_assets is fake_result.assets
        return fake_copy

    def fake_package_campaign(result, output_dir, copy=None):
        call_order.append("package_campaign")
        assert result is fake_result
        assert copy is fake_copy
        return tmp_path / "ecostride_launch"

    monkeypatch.setattr(campaign_service, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign_service.copywriter, "write_campaign_copy", fake_write_campaign_copy)
    monkeypatch.setattr(campaign_service, "package_campaign", fake_package_campaign)

    result, copy, campaign_dir = campaign_service.run_full_campaign(
        "a campaign brief", number_of_assets=3, output_dir=tmp_path
    )

    assert call_order == ["run_campaign", "write_campaign_copy", "package_campaign"]
    assert result is fake_result
    assert copy is fake_copy
    assert campaign_dir == tmp_path / "ecostride_launch"


def test_run_full_campaign_defaults_output_dir_from_settings(monkeypatch, tmp_path):
    fake_result = CampaignResult(brief=_brief(), plan=_plan(), assets=[])
    fake_copy = CampaignCopy(
        campaign_name="EcoStride Launch",
        assets=[
            AssetCopy(
                asset_id="hero", headline="Tread Light", caption="Move fast, tread light.",
                cta="Shop now.", hashtags=["#eco", "#running", "#shoes", "#launch", "#green"],
            )
        ],
    )

    captured = {}

    def fake_run_campaign(text, number_of_assets, output_dir):
        captured["output_dir"] = output_dir
        return fake_result

    monkeypatch.setattr(campaign_service, "run_campaign", fake_run_campaign)
    monkeypatch.setattr(campaign_service.copywriter, "write_campaign_copy", lambda *a, **k: fake_copy)
    monkeypatch.setattr(campaign_service, "package_campaign", lambda *a, **k: tmp_path)
    monkeypatch.setattr(campaign_service.settings, "output_dir", str(tmp_path / "default_outputs"))

    campaign_service.run_full_campaign("a brief")

    assert str(captured["output_dir"]) == str(tmp_path / "default_outputs")
