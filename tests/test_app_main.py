"""Tests for app/main.py -- the CLI entrypoint. run_full_campaign is
mocked; this file's job is argument parsing and exit-code/error handling."""

from __future__ import annotations

from app import main as app_main
from models.exceptions import OllamaUnavailableError
from schemas import (
    AssetCopy,
    AssetPlan,
    AssetType,
    CampaignCopy,
    CampaignPlan,
    CreativeBrief,
    EvaluationResult,
    GeneratedAsset,
    ImagePrompt,
)
from workflows.campaign_workflow import CampaignResult


def _fake_result() -> CampaignResult:
    brief = CreativeBrief(
        campaign_name="EcoStride Launch", campaign_objective="Drive awareness",
        target_audience="College students", key_message="Move fast, tread light.",
        brand_personality=["energetic"], visual_style="bright editorial",
        color_direction=["forest green"], typography_direction="bold sans-serif",
        composition_guidelines="rule of thirds", photography_style="natural light",
        number_of_assets=1, recommended_aspect_ratios=["4:5"],
        content_ideas=["hero shot"], image_generation_strategy="product-in-context",
    )
    plan = CampaignPlan(
        campaign_name="EcoStride Launch",
        assets=[AssetPlan(
            asset_id="hero", asset_type=AssetType.HERO_IMAGE, purpose="Anchor",
            target_audience="College students", message="Statement of values",
            visual_direction="dramatic shot", aspect_ratio="4:5",
            prompt_requirements=["product in focus"],
        )],
    )
    prompt = ImagePrompt(
        positive_prompt="a shoe", negative_prompt="blurry", aspect_ratio="4:5",
        style="editorial", lighting="golden hour", composition="rule of thirds",
        subject="shoe",
    )
    evaluation = EvaluationResult(semantic_score=0.9, technical_score=1.0, overall_score=0.9, passed=True)
    asset = GeneratedAsset(
        asset_id="hero", file_path="outputs/x/images/hero.png", prompt=prompt,
        evaluation=evaluation, attempts_used=1,
    )
    return CampaignResult(brief=brief, plan=plan, assets=[asset])


def test_parse_args_defaults():
    args = app_main.parse_args(["a campaign brief"])
    assert args.brief == "a campaign brief"
    assert args.assets == 5
    assert args.output_dir is None


def test_parse_args_overrides():
    args = app_main.parse_args(["a brief", "--assets", "3", "--output-dir", "/tmp/x"])
    assert args.assets == 3
    assert args.output_dir == "/tmp/x"


def test_main_returns_zero_on_success(monkeypatch, capsys):
    fake_copy = CampaignCopy(
        campaign_name="EcoStride Launch",
        assets=[
            AssetCopy(
                asset_id="hero", headline="Tread Light", caption="Move fast, tread light.",
                cta="Shop now.", hashtags=["#eco", "#running", "#shoes", "#launch", "#green"],
            )
        ],
    )
    fake_result = _fake_result()

    def fake_run_full_campaign(brief, number_of_assets, output_dir):
        return fake_result, fake_copy, "outputs/ecostride_launch"

    monkeypatch.setattr(app_main, "run_full_campaign", fake_run_full_campaign)

    exit_code = app_main.main(["a campaign brief", "--assets", "1"])
    assert exit_code == 0


def test_main_returns_one_on_creative_director_error(monkeypatch):
    def fake_run_full_campaign(brief, number_of_assets, output_dir):
        raise OllamaUnavailableError("Ollama is not reachable at http://localhost:11434")

    monkeypatch.setattr(app_main, "run_full_campaign", fake_run_full_campaign)

    exit_code = app_main.main(["a campaign brief"])
    assert exit_code == 1
