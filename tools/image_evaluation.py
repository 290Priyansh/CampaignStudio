"""
Tool wrapper around evaluation.quality_gate.evaluate_image.

Kept as a thin pass-through, mirroring tools/image_generation.py: agents
and the workflow call this one function, so evaluation internals (which
model, how scores are weighted) can change without touching agent code.
"""

from __future__ import annotations

from evaluation.quality_gate import evaluate_image
from schemas import EvaluationResult, ImagePrompt


def evaluate_asset(
    image_path: str,
    prompt: ImagePrompt,
    *,
    previous_hashes: set[bytes] | None = None,
) -> EvaluationResult:
    return evaluate_image(image_path, prompt, previous_hashes=previous_hashes)
