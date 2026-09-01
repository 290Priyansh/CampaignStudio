"""
Asset generation workflow: the generate -> evaluate -> (revise -> retry)
loop described in sections 15-16 of the project brief, applied to a single
AssetPlan.

Flow per attempt:

    1. Get an ImagePrompt (initial, or revised from prior feedback)
    2. Generate settings.candidates_per_asset candidate images from it
    3. Evaluate every candidate
    4. Rank by overall_score
    5. If the best candidate passes the quality gate -> done
    6. Otherwise, if attempts remain -> revise the prompt using the best
       candidate's feedback and go to step 2
    7. If MAX_GENERATION_ATTEMPTS is exhausted -> return the best candidate
       seen across all attempts, marked as not passed, rather than looping
       forever or raising (a portfolio demo should still show *something*
       for every asset, with its real score visible).

This module intentionally does NOT know about CrewAI/LangGraph -- it's a
plain function operating on this project's own schemas, called by whichever
orchestration layer (workflows/campaign_workflow.py) drives the full
multi-asset campaign.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from agents import prompt_optimizer
from config.settings import settings
from evaluation import hash_file
from schemas import AssetPlan, Candidate, CreativeBrief, GeneratedAsset, ImagePrompt
from tools.image_evaluation import evaluate_asset
from tools.image_generation import ImageGenerator

logger = logging.getLogger("Workflow")


def _aspect_ratio_to_dims(aspect_ratio: str) -> tuple[int, int]:
    """Map an aspect ratio string to concrete pixel dimensions, anchored on
    settings.image_width/height for the longer side. Kept deliberately
    simple (a handful of common Instagram ratios) rather than a general
    solver -- see project brief section 31, 'do not overengineer'."""
    ratio_map = {
        "1:1": (1024, 1024),
        "4:5": (896, 1120),
        "9:16": (768, 1344),
        "16:9": (1344, 768),
    }
    return ratio_map.get(aspect_ratio, (settings.image_width, settings.image_height))


def generate_candidates(
    image_generator: ImageGenerator,
    prompt: ImagePrompt,
    asset: AssetPlan,
    attempt_number: int,
    output_dir: Path,
    previous_hashes: set[bytes],
) -> list[Candidate]:
    """Generate settings.candidates_per_asset candidates for one prompt and
    evaluate each of them."""
    width, height = _aspect_ratio_to_dims(prompt.aspect_ratio)
    candidates: list[Candidate] = []

    for i in range(1, settings.candidates_per_asset + 1):
        candidate_id = f"{asset.asset_id}_a{attempt_number}_c{i}"
        logger.info(
            "Generating candidate %d/%d for asset '%s' (attempt %d/%d)",
            i, settings.candidates_per_asset, asset.asset_id,
            attempt_number, settings.max_generation_attempts,
        )
        out_path = output_dir / f"{candidate_id}_{uuid.uuid4().hex[:8]}.png"

        generated = image_generator.generate(
            prompt.positive_prompt,
            negative_prompt=prompt.negative_prompt,
            width=width,
            height=height,
            output_path=out_path,
            guidance_scale=prompt.guidance_scale,
            num_inference_steps=prompt.num_inference_steps,
        )

        evaluation = evaluate_asset(generated.file_path, prompt, previous_hashes=previous_hashes)
        logger.info("[Evaluator] candidate %s overall_score=%.2f", candidate_id, evaluation.overall_score)
        # evaluate_asset only *compares against* previous_hashes; it doesn't
        # mutate the set, so we add this candidate's hash ourselves before
        # the next candidate (or attempt) is evaluated.
        try:
            previous_hashes.add(hash_file(generated.file_path))
        except OSError as exc:
            logger.warning("Could not hash candidate file %s: %s", generated.file_path, exc)

        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                file_path=generated.file_path,
                prompt=prompt,
                attempt_number=attempt_number,
                evaluation=evaluation,
            )
        )

    return candidates


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Sort candidates best-first by overall_score. Candidates without an
    evaluation (shouldn't normally happen) sort last."""
    return sorted(
        candidates,
        key=lambda c: c.evaluation.overall_score if c.evaluation else -1.0,
        reverse=True,
    )


def generate_asset(
    brief: CreativeBrief,
    asset: AssetPlan,
    image_generator: ImageGenerator,
    output_dir: str | Path,
) -> GeneratedAsset:
    """Run the full generate -> evaluate -> revise -> retry loop for a
    single asset and return the best result found, with a hard cap of
    settings.max_generation_attempts attempts (never an infinite loop)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[Candidate] = []
    previous_hashes: set[bytes] = set()
    prompt = prompt_optimizer.create_initial_prompt(brief, asset)

    for attempt in range(1, settings.max_generation_attempts + 1):
        candidates = generate_candidates(
            image_generator, prompt, asset, attempt, output_dir, previous_hashes
        )
        all_candidates.extend(candidates)

        ranked = rank_candidates(all_candidates)
        best = ranked[0]

        if best.evaluation and best.evaluation.passed:
            logger.info(
                "[Workflow] Asset '%s' passed on attempt %d (score=%.2f)",
                asset.asset_id, attempt, best.evaluation.overall_score,
            )
            return GeneratedAsset(
                asset_id=asset.asset_id,
                file_path=best.file_path,
                prompt=best.prompt,
                evaluation=best.evaluation,
                attempts_used=attempt,
                all_candidates=all_candidates,
            )

        if attempt < settings.max_generation_attempts:
            logger.info(
                "[Workflow] Asset '%s' attempt %d failed quality gate (best score=%.2f) -- revising prompt",
                asset.asset_id, attempt, best.evaluation.overall_score if best.evaluation else 0.0,
            )
            feedback = best.evaluation.feedback if best.evaluation else ["no evaluation available"]
            prompt = prompt_optimizer.revise_prompt(prompt, feedback, brief, asset)

    logger.info(
        "[Workflow] Asset '%s' exhausted %d attempts without passing -- returning best candidate (score=%.2f)",
        asset.asset_id, settings.max_generation_attempts,
        best.evaluation.overall_score if best.evaluation else 0.0,
    )
    return GeneratedAsset(
        asset_id=asset.asset_id,
        file_path=best.file_path,
        prompt=best.prompt,
        evaluation=best.evaluation,
        attempts_used=settings.max_generation_attempts,
        all_candidates=all_candidates,
    )
