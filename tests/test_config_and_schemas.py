"""
Phase 1 tests: prove the config layer loads sane defaults and every schema
validates real-shaped data. No LLM, no GPU, no network required.
"""

import pytest
from pydantic import ValidationError

from config.settings import Settings
from schemas import (
    AssetPlan,
    AssetType,
    AudienceProfile,
    CampaignPlan,
    Candidate,
    CreativeBrief,
    EvaluationResult,
    GeneratedAsset,
    ImagePrompt,
)


def test_settings_defaults_are_zero_cost_and_local():
    s = Settings(_env_file=None)  # ignore any local .env during tests
    assert s.ollama_base_url.startswith("http://localhost")
    assert s.image_backend in ("diffusers", "comfyui")
    assert s.max_generation_attempts >= 1
    assert 0.0 <= s.quality_threshold <= 1.0
    # No field on Settings should reference a paid API key.
    forbidden = {"openai_api_key", "anthropic_api_key", "gemini_api_key", "replicate_api_key"}
    assert forbidden.isdisjoint(s.model_fields.keys())


def test_settings_model_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:4b")
    monkeypatch.setenv("QUALITY_THRESHOLD", "0.9")
    s = Settings(_env_file=None)
    assert s.ollama_model == "gemma3:4b"
    assert s.quality_threshold == 0.9


def test_creative_brief_round_trip():
    brief = CreativeBrief(
        campaign_name="EcoStride Launch",
        campaign_objective="Drive awareness for a new eco-friendly running shoe.",
        target_audience="College students, 18-24",
        key_message="Move fast, tread light.",
        brand_personality=["energetic", "modern", "sustainable"],
        visual_style="bright, high-contrast editorial photography",
        color_direction=["forest green", "off-white", "sunset orange"],
        typography_direction="bold geometric sans-serif",
        composition_guidelines="rule of thirds, room for text overlay",
        photography_style="natural light documentary",
        number_of_assets=5,
        recommended_aspect_ratios=["1:1", "4:5", "9:16"],
        content_ideas=["hero shoe shot", "campus lifestyle", "sustainability close-up"],
        image_generation_strategy="prioritize product-in-context over studio isolates",
    )
    dumped = brief.model_dump_json()
    restored = CreativeBrief.model_validate_json(dumped)
    assert restored.campaign_name == "EcoStride Launch"
    assert restored.audience_profile is None


def test_creative_brief_rejects_invalid_asset_count():
    with pytest.raises(ValidationError):
        CreativeBrief(
            campaign_name="X",
            campaign_objective="X",
            target_audience="X",
            key_message="X",
            brand_personality=["x"],
            visual_style="x",
            color_direction=["x"],
            typography_direction="x",
            composition_guidelines="x",
            photography_style="x",
            number_of_assets=0,  # invalid: must be >= 1
            recommended_aspect_ratios=["1:1"],
            content_ideas=["x"],
            image_generation_strategy="x",
        )


def test_audience_profile_requires_non_empty_lists():
    with pytest.raises(ValidationError):
        AudienceProfile(
            demographic_summary="x",
            motivations=[],  # invalid: min_length=1
            pain_points=["x"],
            desired_emotional_response="x",
            platform_behavior="x",
            recommended_tone="x",
        )


def test_campaign_plan_and_asset_plan():
    plan = CampaignPlan(
        campaign_name="EcoStride Launch",
        assets=[
            AssetPlan(
                asset_id="hero",
                asset_type=AssetType.HERO_IMAGE,
                purpose="Anchor visual for the launch",
                target_audience="College students",
                message="The shoe as a statement of values",
                visual_direction="dramatic side-lit product shot",
                aspect_ratio="4:5",
                prompt_requirements=["product in focus", "green/orange accents"],
            )
        ],
    )
    assert plan.assets[0].asset_type == AssetType.HERO_IMAGE


def test_image_prompt_revision_preserves_fields_and_annotates():
    prompt = ImagePrompt(
        positive_prompt="a running shoe on a forest trail, golden hour",
        negative_prompt="blurry, watermark, extra limbs",
        aspect_ratio="4:5",
        style="editorial photography",
        lighting="golden hour",
        composition="rule of thirds",
        subject="eco-friendly running shoe",
    )
    revised = prompt.revised(
        feedback=["subject too small in frame"],
        revision_note="increase subject scale, move to foreground",
    )
    assert revised.aspect_ratio == prompt.aspect_ratio
    assert "revision:" in revised.positive_prompt
    assert revised.positive_prompt != prompt.positive_prompt


def test_evaluation_result_overall_score_bounds():
    with pytest.raises(ValidationError):
        EvaluationResult(
            semantic_score=0.5,
            technical_score=0.5,
            overall_score=1.5,  # invalid: must be <= 1.0
            passed=False,
        )


def test_generated_asset_composes_candidates_and_evaluation():
    prompt = ImagePrompt(
        positive_prompt="p",
        negative_prompt="n",
        aspect_ratio="1:1",
        style="s",
        lighting="l",
        composition="c",
        subject="subj",
    )
    passing_eval = EvaluationResult(
        semantic_score=0.8, technical_score=0.9, overall_score=0.85, passed=True
    )
    candidate = Candidate(
        candidate_id="hero_c1", file_path="outputs/x/images/hero_c1.png",
        prompt=prompt, attempt_number=1, evaluation=passing_eval,
    )
    asset = GeneratedAsset(
        asset_id="hero", file_path=candidate.file_path, prompt=prompt,
        evaluation=passing_eval, attempts_used=1, all_candidates=[candidate],
    )
    assert asset.evaluation.passed is True
    assert asset.all_candidates[0].candidate_id == "hero_c1"
