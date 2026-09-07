"""
Creative Director agent (CrewAI).

Produces the top-level CreativeBrief from a raw campaign-brief string. This
is the entry point of the whole system -- everything downstream (Audience
Analyst, Campaign Strategist, Prompt Optimizer, Copywriter) is grounded in
the CreativeBrief this agent returns.

Uses CrewAI's own structured-output mechanism (`Task(output_pydantic=...)`)
rather than `models.llm.invoke_structured`, because this agent's job is
role-played creative reasoning driven by CrewAI's Agent/Task/Crew
machinery. The Prompt Optimizer (agents/prompt_optimizer.py) uses
LangChain's `with_structured_output` instead, because that agent isn't
"role-playing" a persona -- it's a narrower, mechanical transformation
where guaranteed schema compliance matters more than persona-driven
reasoning. See README 'Tech Stack' for the full explanation of this split.

CrewAI is imported locally (inside functions) so this module -- and its
prompt-assembly logic -- can be unit-tested without crewai installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.exceptions import CrewExecutionError
from models.llm import execute_agent_task, get_crewai_llm
from schemas import CreativeBrief

logger = logging.getLogger("CreativeDirector")

_BACKSTORY_PATH = Path(__file__).resolve().parent.parent / "prompts" / "creative_director.txt"


def _load_backstory() -> str:
    return _BACKSTORY_PATH.read_text(encoding="utf-8")


def build_agent():
    from crewai import Agent

    return Agent(
        role="Creative Director",
        goal=(
            "Translate a high-level campaign brief into a complete, structured "
            "creative direction that every downstream specialist can execute "
            "against without further clarification."
        ),
        backstory=_load_backstory(),
        llm=get_crewai_llm(),
        verbose=False,
        allow_delegation=False,
    )


def _build_task(agent, campaign_brief_text: str, number_of_assets: int):
    from crewai import Task

    return Task(
        description=(
            f"Campaign brief from the client:\n\n{campaign_brief_text}\n\n"
            f"Produce a complete creative brief for a {number_of_assets}-asset "
            "Instagram campaign based on this input. Be concrete and specific -- "
            "every field will be used directly by other specialists (an "
            "audience analyst, a campaign strategist, and an image prompt "
            "engineer) with no further access to the original brief text."
        ),
        expected_output="A complete, structured creative brief.",
        agent=agent,
        output_pydantic=CreativeBrief,
    )


def create_brief(campaign_brief_text: str, number_of_assets: int = 5) -> CreativeBrief:
    """Run the Creative Director agent on a raw campaign brief string and
    return a validated CreativeBrief."""
    from crewai import Crew, Process

    agent = build_agent()
    task = _build_task(agent, campaign_brief_text, number_of_assets)

    logger.info("Creating campaign strategy from brief: %.80s...", campaign_brief_text)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    prompt = task.description
    brief = execute_agent_task("Creative Director", crew, task, CreativeBrief, prompt)
    brief.number_of_assets = number_of_assets
    return brief

