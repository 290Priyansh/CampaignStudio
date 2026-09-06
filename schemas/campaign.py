"""Campaign-plan-level schemas -- output of the Campaign Strategist agent."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AssetType(str, Enum):
    HERO_IMAGE = "hero_image"
    PRODUCT_IMAGE = "product_image"
    LIFESTYLE_IMAGE = "lifestyle_image"
    PROMOTIONAL_POSTER = "promotional_poster"
    INSTAGRAM_STORY = "instagram_story"


class AssetPlan(BaseModel):
    """One planned asset within a campaign, before any prompt/image exists."""

    asset_id: str = Field(description="Short slug, e.g. 'hero', 'lifestyle_01'.")
    asset_type: AssetType
    purpose: str = Field(description="Why this asset exists in the campaign.")
    target_audience: str
    message: str = Field(description="The specific message this single asset conveys.")
    visual_direction: str
    aspect_ratio: str = Field(description="e.g. '1:1', '4:5', '9:16'.")
    prompt_requirements: list[str] = Field(
        description="Concrete elements the image prompt must include.",
        min_length=1,
    )

    @field_validator("asset_type", mode="before")
    @classmethod
    def normalize_asset_type(cls, v: str | AssetType) -> AssetType:
        if isinstance(v, str):
            v_clean = v.lower().strip()
            valid_values = {e.value for e in AssetType}
            if v_clean in valid_values:
                return AssetType(v_clean)
            if "hero" in v_clean or "header" in v_clean:
                return AssetType.HERO_IMAGE
            if "story" in v_clean or "reels" in v_clean:
                return AssetType.INSTAGRAM_STORY
            if "product" in v_clean or "packshot" in v_clean:
                return AssetType.PRODUCT_IMAGE
            if "poster" in v_clean or "banner" in v_clean or "flyer" in v_clean:
                return AssetType.PROMOTIONAL_POSTER
            if "lifestyle" in v_clean or "social" in v_clean or "ad" in v_clean:
                return AssetType.LIFESTYLE_IMAGE
            return AssetType.LIFESTYLE_IMAGE
        return v


class CampaignPlan(BaseModel):
    """Full plan: a set of AssetPlans derived from a CreativeBrief."""

    campaign_name: str
    assets: list[AssetPlan] = Field(min_length=1)
