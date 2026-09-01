"""
Copywriter agent (CrewAI).

Per section 18 of the project brief, this is a downstream component that
runs *after* visual assets have been generated and selected -- it is the
direct descendant of the original repo's Content Writer/Reviewer agents,
but now grounded in the real, final AssetPlan + GeneratedAsset (including
the actual image prompt that was used), not just the raw campaign topic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.exceptions import CrewExecutionError
from models.llm import get_crewai_llm
from schemas import AssetPlan, CampaignCopy, CreativeBrief, GeneratedAsset

logger = logging.getLogger("Copywriter")

_BACKSTORY_PATH = Path(__file__).resolve().parent.parent / "prompts" / "copywriter.txt"


def _load_backstory() -> str:
    return _BACKSTORY_PATH.read_text()


def build_agent():
    from crewai import Agent

    return Agent(
        role="Social Copywriter",
        goal=(
            "Write Instagram copy for each generated asset that is specific "
            "to what that asset actually shows and why it exists in this "
            "campaign -- never generic, interchangeable filler."
        ),
        backstory=_load_backstory(),
        llm=get_crewai_llm(),
        verbose=False,
        allow_delegation=False,
    )


def _describe_asset(asset_plan: AssetPlan, generated: GeneratedAsset) -> str:
    return (
        f"- asset_id: {generated.asset_id} ({asset_plan.asset_type.value})\n"
        f"  purpose: {asset_plan.purpose}\n"
        f"  message: {asset_plan.message}\n"
        f"  what the image actually depicts (its generation prompt): "
        f"{generated.prompt.positive_prompt}\n"
        f"  quality score achieved: {generated.evaluation.overall_score:.2f}"
    )


def _build_task(agent, brief: CreativeBrief, asset_plans: list[AssetPlan], generated_assets: list[GeneratedAsset]):
    from crewai import Task

    plans_by_id = {p.asset_id: p for p in asset_plans}
    asset_descriptions = "\n".join(
        _describe_asset(plans_by_id[g.asset_id], g)
        for g in generated_assets
        if g.asset_id in plans_by_id
    )

    return Task(
        description=(
            f"Campaign: {brief.campaign_name}\n"
            f"Key message: {brief.key_message}\n"
            f"Brand personality: {', '.join(brief.brand_personality)}\n"
            f"Recommended tone: "
            f"{brief.audience_profile.recommended_tone if brief.audience_profile else 'not specified'}\n\n"
            f"Assets produced for this campaign:\n{asset_descriptions}\n\n"
            "Write a headline, full caption, call-to-action, and 5-15 hashtags "
            "for EVERY asset listed above, using its asset_id exactly as given. "
            "Ground each caption in what that specific asset depicts -- do not "
            "write copy that could be swapped between assets without anyone "
            "noticing."
        ),
        expected_output=f"Campaign copy for all {len(generated_assets)} assets.",
        agent=agent,
        output_pydantic=CampaignCopy,
    )


def write_campaign_copy(
    brief: CreativeBrief,
    asset_plans: list[AssetPlan],
    generated_assets: list[GeneratedAsset],
) -> CampaignCopy:
    """Run the Copywriter agent and return validated CampaignCopy."""
    from crewai import Crew, Process

    agent = build_agent()
    task = _build_task(agent, brief, asset_plans, generated_assets)

    logger.info("Writing campaign copy for '%s' (%d assets)", brief.campaign_name, len(generated_assets))
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    try:
        crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        raise CrewExecutionError(f"Copywriter crew run failed: {exc}") from exc

    result = getattr(task.output, "pydantic", None)
    if result is None:
        raise CrewExecutionError(
            "Copywriter agent did not return a valid structured CampaignCopy"
        )
    return result
