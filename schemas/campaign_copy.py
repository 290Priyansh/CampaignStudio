"""
Social copy schema -- this is the direct descendant of the original repo's
Content Writer / Reviewer agents, now a downstream component that runs
*after* a visual asset has been selected, per section 18 of the brief.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AssetCopy(BaseModel):
    asset_id: str = Field(description="Matches GeneratedAsset.asset_id.")
    headline: str
    caption: str = Field(description="Full Instagram-ready caption, 2-3 short paragraphs.")
    cta: str
    hashtags: list[str] = Field(min_length=5, max_length=15)


class CampaignCopy(BaseModel):
    campaign_name: str
    assets: list[AssetCopy] = Field(min_length=1)
