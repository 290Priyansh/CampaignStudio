"""
Top-level campaign service -- the single entry point the CLI/UI call.

Chains together everything built in earlier phases:

    workflows.campaign_workflow.run_campaign()   -> CampaignResult
    agents.copywriter.write_campaign_copy()      -> CampaignCopy
    services.asset_service.package_campaign()    -> Path to final package

This is intentionally the *only* place these three are wired together, so
app/main.py and ui/streamlit_app.py both just call `run_full_campaign()`
rather than duplicating orchestration logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agents import copywriter
from config.settings import settings
from schemas import CampaignCopy
from services.asset_service import package_campaign
from workflows.campaign_workflow import CampaignResult, run_campaign

logger = logging.getLogger("Workflow")


def run_full_campaign(
    campaign_brief_text: str,
    number_of_assets: int = 5,
    output_dir: str | Path | None = None,
) -> tuple[CampaignResult, CampaignCopy, Path]:
    """Run the full pipeline end to end: strategy -> generation/evaluation
    -> copywriting -> packaging. Returns the raw result objects plus the
    path to the packaged campaign folder, so callers (UI/CLI) can display
    either the structured data or just point the user at the folder."""
    output_dir = Path(output_dir) if output_dir else Path(settings.output_dir)

    result = run_campaign(campaign_brief_text, number_of_assets=number_of_assets, output_dir=output_dir)

    logger.info("[Copywriter] Writing campaign copy")
    copy = copywriter.write_campaign_copy(result.brief, result.plan.assets, result.assets)

    campaign_dir = package_campaign(result, output_dir, copy=copy)

    return result, copy, campaign_dir
