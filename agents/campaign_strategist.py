"""
Campaign Strategist agent (CrewAI).

Converts a CreativeBrief (with its AudienceProfile already attached) into a
CampaignPlan: a concrete list of AssetPlans, each sized and scoped for its
actual placement (project brief section 11). This is what the workflow
loops over, calling workflows.asset_generation.generate_asset() once per
AssetPlan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.exceptions import CrewExecutionError
from models.llm import get_crewai_llm
from schemas import CampaignPlan, CreativeBrief

logger = logging.getLogger("CampaignStrategist")

_BACKSTORY_PATH = Path(__file__).resolve().parent.parent / "prompts" / "campaign_strategist.txt"


def _load_backstory() -> str:
    return _BACKSTORY_PATH.read_text()


def build_agent():
    from crewai import Agent

    return Agent(
        role="Campaign Strategist",
        goal=(
            "Convert a creative brief into a concrete, minimal set of assets "
            "that together fulfil the campaign's objective -- each one sized "
            "and scoped for its actual placement."
        ),
        backstory=_load_backstory(),
        llm=get_crewai_llm(),
        verbose=False,
        allow_delegation=False,
    )


def _build_task(agent, brief: CreativeBrief):
    from crewai import Task

    audience_context = ""
    if brief.audience_profile:
        ap = brief.audience_profile
        audience_context = (
            f"\nAudience insight: {ap.demographic_summary}\n"
            f"Desired emotional response: {ap.desired_emotional_response}\n"
            f"Recommended tone: {ap.recommended_tone}\n"
            f"Platform behavior: {ap.platform_behavior}\n"
        )

    return Task(
        description=(
            f"Campaign: {brief.campaign_name}\n"
            f"Objective: {brief.campaign_objective}\n"
            f"Key message: {brief.key_message}\n"
            f"Visual style: {brief.visual_style}\n"
            f"Content ideas from the Creative Director: {'; '.join(brief.content_ideas)}\n"
            f"Recommended aspect ratios: {', '.join(brief.recommended_aspect_ratios)}\n"
            f"{audience_context}\n"
            f"Plan exactly {brief.number_of_assets} assets for this campaign. "
            "For each one, specify its purpose, its message, its visual "
            "direction, its aspect ratio, and the concrete elements its "
            "image prompt must include. A prompt engineer will execute your "
            "plan with no further creative judgment calls -- be specific."
        ),
        expected_output=f"A campaign plan with exactly {brief.number_of_assets} planned assets.",
        agent=agent,
        output_pydantic=CampaignPlan,
    )


def build_campaign_plan(brief: CreativeBrief) -> CampaignPlan:
    """Run the Campaign Strategist agent and return a validated CampaignPlan."""
    from crewai import Crew, Process

    agent = build_agent()
    task = _build_task(agent, brief)

    logger.info("Building campaign plan for '%s' (%d assets)", brief.campaign_name, brief.number_of_assets)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    try:
        crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        raise CrewExecutionError(f"Campaign Strategist crew run failed: {exc}") from exc

    result = getattr(task.output, "pydantic", None)
    if result is None:
        raise CrewExecutionError(
            "Campaign Strategist agent did not return a valid structured CampaignPlan"
        )
    return result
