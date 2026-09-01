"""
Phase 5 tests for agents/prompt_optimizer.py. The LLM call
(models.llm.invoke_structured) is mocked -- these tests verify the prompt
template is filled correctly and that revision carries prior-prompt +
feedback context, not that a real local model produces good prompts (that
requires Ollama running, which is a manual/integration-level check -- see
README).
"""

from __future__ import annotations

from agents import prompt_optimizer
from schemas import AssetPlan, AssetType, CreativeBrief, ImagePrompt


def _make_brief() -> CreativeBrief:
    return CreativeBrief(
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
        content_ideas=["hero shoe shot"],
        image_generation_strategy="prioritize product-in-context over studio isolates",
    )


def _make_asset() -> AssetPlan:
    return AssetPlan(
        asset_id="hero",
        asset_type=AssetType.HERO_IMAGE,
        purpose="Anchor visual for the launch",
        target_audience="College students",
        message="The shoe as a statement of values",
        visual_direction="dramatic side-lit product shot",
        aspect_ratio="4:5",
        prompt_requirements=["product in focus", "green/orange accents"],
    )


def _make_image_prompt(text="a running shoe, editorial photography") -> ImagePrompt:
    return ImagePrompt(
        positive_prompt=text,
        negative_prompt="blurry, watermark",
        aspect_ratio="4:5",
        style="editorial",
        lighting="golden hour",
        composition="rule of thirds",
        subject="running shoe",
    )


def test_create_initial_prompt_fills_template_and_calls_llm(monkeypatch):
    captured = {}

    def fake_invoke_structured(schema, prompt_text):
        captured["schema"] = schema
        captured["prompt_text"] = prompt_text
        return _make_image_prompt()

    monkeypatch.setattr(prompt_optimizer, "invoke_structured", fake_invoke_structured)

    result = prompt_optimizer.create_initial_prompt(_make_brief(), _make_asset())

    assert captured["schema"] is ImagePrompt
    assert "EcoStride Launch" in captured["prompt_text"]
    assert "hero_image" in captured["prompt_text"]
    assert "dramatic side-lit product shot" in captured["prompt_text"]
    assert "forest green" in captured["prompt_text"]
    assert "(none -- this is the first attempt)" in captured["prompt_text"]
    assert isinstance(result, ImagePrompt)


def test_revise_prompt_includes_previous_prompt_and_feedback(monkeypatch):
    captured = {}

    def fake_invoke_structured(schema, prompt_text):
        captured["prompt_text"] = prompt_text
        return _make_image_prompt(text="revised prompt text")

    monkeypatch.setattr(prompt_optimizer, "invoke_structured", fake_invoke_structured)

    previous = _make_image_prompt(text="original prompt text")
    feedback = ["subject too small in frame", "low sharpness detected"]

    result = prompt_optimizer.revise_prompt(previous, feedback, _make_brief(), _make_asset())

    assert "original prompt text" in captured["prompt_text"]
    assert "subject too small in frame" in captured["prompt_text"]
    assert "low sharpness detected" in captured["prompt_text"]
    assert result.positive_prompt == "revised prompt text"
