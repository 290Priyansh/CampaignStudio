"""
Top-level campaign workflow.

This is the orchestration described in the project's architecture diagram:

    Campaign brief (string)
        -> Creative Director        -> CreativeBrief
        -> Audience Analyst         -> AudienceProfile (attached to brief)
        -> Campaign Strategist      -> CampaignPlan (list of AssetPlans)
        -> for each AssetPlan:
               workflows.asset_generation.generate_asset()
                   -> Prompt Optimizer -> ImageGenerator -> Evaluator
                   -> (revise/retry up to MAX_GENERATION_ATTEMPTS)
        -> CampaignResult (CreativeBrief + CampaignPlan + GeneratedAssets)

Copywriting (per section 18 of the brief, a downstream step that runs
*after* a visual asset is selected) and output packaging (section 19) are
separate, later pieces -- this module's job stops at "campaign fully
generated and evaluated."
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from agents import audience_analyst, campaign_strategist, creative_director
from config.settings import settings
from models.llm import check_ollama_available
from models.exceptions import OllamaUnavailableError
from schemas import CampaignPlan, CreativeBrief, GeneratedAsset
from tools.image_generation import get_image_generator
from workflows.asset_generation import generate_asset

logger = logging.getLogger("Workflow")


class CampaignResult(BaseModel):
    brief: CreativeBrief
    plan: CampaignPlan
    assets: list[GeneratedAsset]


def run_campaign(
    campaign_brief_text: str,
    number_of_assets: int = 5,
    output_dir: str | Path | None = None,
) -> CampaignResult:
    """Run the full campaign pipeline end to end and return a CampaignResult.

    Fails fast with a clear OllamaUnavailableError if Ollama isn't reachable,
    rather than letting the first agent call time out deep inside a Crew run.
    """
    if not check_ollama_available():
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {settings.ollama_base_url}. Start it with "
            f"`ollama serve` and pull the configured model with "
            f"`ollama pull {settings.ollama_model}` before running a campaign."
        )

    output_dir = Path(output_dir) if output_dir else Path(settings.output_dir)

    logger.info("[CreativeDirector] Creating campaign strategy")
    brief = creative_director.create_brief(campaign_brief_text, number_of_assets=number_of_assets)

    logger.info("[AudienceAnalyst] Analyzing target audience")
    brief.audience_profile = audience_analyst.analyze_audience(brief)

    logger.info("[CampaignStrategist] Planning campaign assets")
    plan = campaign_strategist.build_campaign_plan(brief)

    image_generator = get_image_generator()
    campaign_output_dir = output_dir / _slugify(brief.campaign_name) / "images"

    generated_assets: list[GeneratedAsset] = []
    for asset_plan in plan.assets:
        logger.info("[Workflow] Generating asset '%s' (%s)", asset_plan.asset_id, asset_plan.asset_type.value)
        generated = generate_asset(brief, asset_plan, image_generator, campaign_output_dir)
        generated_assets.append(generated)

    return CampaignResult(brief=brief, plan=plan, assets=generated_assets)


def _slugify(name: str) -> str:
    from tools.file_tools import slugify

    return slugify(name)
