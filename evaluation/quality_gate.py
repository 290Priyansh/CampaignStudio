"""
Combines semantic (CLIP) + aesthetic (optional) + technical scores into a
single EvaluationResult and applies `settings.quality_threshold`.

This is the one function the workflow calls after every generation
attempt (project brief sections 14-15). If aesthetic scoring is
unavailable (no weights configured), its weight is redistributed across
semantic + technical rather than silently shrinking the achievable score.
"""

from __future__ import annotations

import logging

from config.settings import settings
from evaluation.aesthetic_evaluator import AestheticEvaluator
from evaluation.clip_evaluator import CLIPEvaluator
from evaluation.technical_checks import run_technical_checks
from schemas import EvaluationResult, ImagePrompt

logger = logging.getLogger("Evaluator")

SEMANTIC_WEIGHT = 0.5
AESTHETIC_WEIGHT = 0.2
TECHNICAL_WEIGHT = 0.3

# Module-level singletons so the (potentially slow) CLIP model is loaded
# once per process, not once per evaluate_image() call.
_clip_evaluator = CLIPEvaluator()
_aesthetic_evaluator = AestheticEvaluator(clip_evaluator=_clip_evaluator)


def evaluate_image(
    image_path: str,
    prompt: ImagePrompt,
    *,
    previous_hashes: set[bytes] | None = None,
    threshold: float | None = None,
) -> EvaluationResult:
    threshold = threshold if threshold is not None else settings.quality_threshold

    technical_score, feedback, _hash = run_technical_checks(
        image_path, previous_hashes=previous_hashes
    )
    feedback = list(feedback)

    semantic_score = 0.0
    try:
        semantic_score = _clip_evaluator.score(image_path, prompt.positive_prompt)
    except Exception as exc:  # noqa: BLE001 - scoring failure shouldn't crash the workflow
        logger.warning("CLIP scoring failed: %s", exc)
        feedback.append(f"semantic scoring failed: {exc}")

    aesthetic_score: float | None = None
    try:
        aesthetic_score = _aesthetic_evaluator.score(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Aesthetic scoring failed: %s", exc)

    if aesthetic_score is None:
        weight_sum = SEMANTIC_WEIGHT + TECHNICAL_WEIGHT
        overall = (semantic_score * SEMANTIC_WEIGHT + technical_score * TECHNICAL_WEIGHT) / weight_sum
    else:
        overall = (
            semantic_score * SEMANTIC_WEIGHT
            + aesthetic_score * AESTHETIC_WEIGHT
            + technical_score * TECHNICAL_WEIGHT
        )

    passed = overall >= threshold

    if semantic_score < 0.5:
        feedback.append(
            "generated image has low semantic similarity to the prompt's intended subject"
        )
    if not passed:
        feedback.append(f"overall score {overall:.2f} below quality threshold {threshold:.2f}")

    logger.info(
        "semantic=%.2f aesthetic=%s technical=%.2f overall=%.2f threshold=%.2f passed=%s",
        semantic_score,
        f"{aesthetic_score:.2f}" if aesthetic_score is not None else "n/a",
        technical_score,
        overall,
        threshold,
        passed,
    )

    return EvaluationResult(
        semantic_score=semantic_score,
        aesthetic_score=aesthetic_score,
        technical_score=technical_score,
        overall_score=round(overall, 4),
        passed=passed,
        feedback=feedback,
    )
