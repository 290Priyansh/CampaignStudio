"""
Structured output for the Creative Director agent.

Replaces the original repo's pattern of regex-scraping ```json fences out of
a raw LLM string. Every agent in this system returns (or is coerced into)
one of these Pydantic models, so downstream agents/tools get a stable
contract instead of hoping the LLM's JSON formatting cooperates.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudienceProfile(BaseModel):
    """Structured recommendations from the Audience Analyst agent.

    This is *not* meant to be a generic paragraph -- every field here should
    directly influence downstream copy or visual direction.
    """

    demographic_summary: str = Field(
        description="1-2 sentence description of the target demographic (age, "
        "life stage, context)."
    )
    motivations: list[str] = Field(
        description="What this audience wants / values, 3-5 short items.",
        min_length=1,
    )
    pain_points: list[str] = Field(
        description="Problems or frustrations this campaign should speak to, 2-4 items."
    )
    desired_emotional_response: str = Field(
        description="The single emotion the campaign should evoke (e.g. 'aspirational "
        "confidence', 'nostalgic warmth')."
    )
    platform_behavior: str = Field(
        description="How this audience actually consumes Instagram content "
        "(scroll speed, format preference, attention span)."
    )
    recommended_tone: str = Field(
        description="Tone of voice recommendation for copy (e.g. 'playful but "
        "credible', 'minimal and confident')."
    )


class CreativeBrief(BaseModel):
    """Top-level structured brief produced by the Creative Director agent.

    This is the single artifact that the rest of the pipeline (Campaign
    Strategist, Prompt Optimizer, Copywriter) is grounded in.
    """

    campaign_name: str
    campaign_objective: str = Field(
        description="What business/marketing outcome this campaign serves."
    )
    target_audience: str = Field(
        description="Short description; full detail lives in AudienceProfile."
    )
    key_message: str = Field(description="The single core message of the campaign.")
    brand_personality: list[str] = Field(
        description="3-5 adjectives describing brand voice (e.g. 'energetic', "
        "'modern', 'sustainable')."
    )
    visual_style: str = Field(
        description="Overall visual direction, e.g. 'bright, high-contrast, "
        "editorial photography'."
    )
    color_direction: list[str] = Field(
        description="Named colors or hex-ish descriptors driving the palette.",
        min_length=1,
    )
    typography_direction: str = Field(
        description="Typographic feel, even though this system doesn't render "
        "type -- it informs mood/style language passed to the image prompts."
    )
    composition_guidelines: str = Field(
        description="Framing/composition preferences, e.g. 'rule of thirds, "
        "negative space for text overlay'."
    )
    photography_style: str = Field(
        description="e.g. 'natural light documentary photography' or "
        "'stylized 3D render'."
    )
    number_of_assets: int = Field(ge=1, le=10)
    recommended_aspect_ratios: list[str] = Field(
        description="e.g. ['1:1', '4:5', '9:16']", min_length=1
    )
    content_ideas: list[str] = Field(
        description="Short list of concrete content/asset ideas.", min_length=1
    )
    image_generation_strategy: str = Field(
        description="High-level notes for the Prompt Optimizer, e.g. 'prioritize "
        "product-in-context shots over studio isolates'."
    )
    audience_profile: AudienceProfile | None = Field(
        default=None,
        description="Populated once the Audience Analyst agent has run.",
    )
