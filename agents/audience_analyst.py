"""
Audience Analyst agent (CrewAI).

Takes a CreativeBrief (already produced by the Creative Director) and
returns a structured AudienceProfile -- concrete recommendations that
directly influence downstream copy and visual direction, not a generic
research paragraph (project brief section 10 is explicit about this).
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.exceptions import CrewExecutionError
from models.llm import get_crewai_llm
from schemas import AudienceProfile, CreativeBrief

logger = logging.getLogger("AudienceAnalyst")

_BACKSTORY_PATH = Path(__file__).resolve().parent.parent / "prompts" / "audience_analyst.txt"


def _load_backstory() -> str:
    return _BACKSTORY_PATH.read_text()


def build_agent():
    from crewai import Agent

    return Agent(
        role="Audience Analyst",
        goal=(
            "Turn a creative brief's target audience description into concrete, "
            "usable recommendations that change what the creative team actually "
            "produces -- not generic demographic commentary."
        ),
        backstory=_load_backstory(),
        llm=get_crewai_llm(),
        verbose=False,
        allow_delegation=False,
    )


def _build_task(agent, brief: CreativeBrief):
    from crewai import Task

    return Task(
        description=(
            f"Campaign: {brief.campaign_name}\n"
            f"Objective: {brief.campaign_objective}\n"
            f"Target audience (as given): {brief.target_audience}\n"
            f"Key message: {brief.key_message}\n"
            f"Brand personality: {', '.join(brief.brand_personality)}\n\n"
            "Analyze this audience specifically for an Instagram campaign. "
            "Every field you return must be specific enough to change a "
            "creative decision -- if a statement could apply to almost any "
            "audience, rewrite it until it's specific to this one."
        ),
        expected_output="A complete, structured audience profile.",
        agent=agent,
        output_pydantic=AudienceProfile,
    )


def analyze_audience(brief: CreativeBrief) -> AudienceProfile:
    """Run the Audience Analyst agent and return a validated AudienceProfile."""
    from crewai import Crew, Process

    agent = build_agent()
    task = _build_task(agent, brief)

    logger.info("Analyzing target audience for campaign '%s'", brief.campaign_name)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    try:
        crew.kickoff()
    except Exception as exc:  # noqa: BLE001
        raise CrewExecutionError(f"Audience Analyst crew run failed: {exc}") from exc

    result = getattr(task.output, "pydantic", None)
    if result is None:
        raise CrewExecutionError(
            "Audience Analyst agent did not return a valid structured AudienceProfile"
        )
    return result
