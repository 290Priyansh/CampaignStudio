"""
Prompt Optimizer agent.

Converts an AssetPlan (+ CreativeBrief context) into a structured
ImagePrompt ready for the ImageGenerator tool, and revises that prompt
using evaluator feedback when a candidate fails the quality gate (see
workflows/campaign_workflow.py, next checkpoint, for how this loop is
driven end-to-end).

This is the concrete implementation of section 12 of the project brief:
the LLM never goes straight from "campaign description" to "Stable
Diffusion call". It's always routed through explicit reasoning about
subject/composition/lighting/etc. and returns a validated ImagePrompt via
LangChain structured output (models.llm.invoke_structured) -- not a
hand-parsed string, which was the original repo's approach.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.llm import invoke_structured
from schemas import AssetPlan, CreativeBrief, ImagePrompt

logger = logging.getLogger("PromptOptimizer")

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "prompt_optimizer.txt"


def _load_template() -> str:
    return _TEMPLATE_PATH.read_text()


def _fill_template(brief: CreativeBrief, asset: AssetPlan, feedback_section: str) -> str:
    return _load_template().format(
        campaign_name=brief.campaign_name,
        visual_style=brief.visual_style,
        color_direction=", ".join(brief.color_direction),
        photography_style=brief.photography_style,
        composition_guidelines=brief.composition_guidelines,
        asset_type=asset.asset_type.value,
        asset_purpose=asset.purpose,
        message=asset.message,
        visual_direction=asset.visual_direction,
        aspect_ratio=asset.aspect_ratio,
        prompt_requirements="; ".join(asset.prompt_requirements),
        feedback=feedback_section,
    )


def create_initial_prompt(brief: CreativeBrief, asset: AssetPlan) -> ImagePrompt:
    """Produce the first-attempt ImagePrompt for a given asset plan."""
    filled = _fill_template(brief, asset, feedback_section="(none -- this is the first attempt)")
    logger.info("Creating initial image prompt for asset '%s'", asset.asset_id)
    return invoke_structured(ImagePrompt, filled)


def revise_prompt(
    previous: ImagePrompt,
    feedback: list[str],
    brief: CreativeBrief,
    asset: AssetPlan,
) -> ImagePrompt:
    """Produce a revised ImagePrompt informed by evaluator feedback."""
    feedback_section = (
        f"Previous positive prompt: {previous.positive_prompt}\n"
        f"Previous negative prompt: {previous.negative_prompt}\n"
        "Evaluator feedback to address in this revision: " + "; ".join(feedback)
    )
    filled = _fill_template(brief, asset, feedback_section=feedback_section)
    logger.info("Revising image prompt for asset '%s' based on feedback: %s", asset.asset_id, feedback)
    return invoke_structured(ImagePrompt, filled)
