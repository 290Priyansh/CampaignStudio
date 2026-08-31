"""Schemas for generated images, their evaluation, and final selected assets."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.image_prompt import ImagePrompt


class EvaluationResult(BaseModel):
    """Output of the evaluation stage (tools/image_evaluation.py + evaluation/*).

    semantic_score:  CLIP image<->prompt similarity, normalized 0-1.
    aesthetic_score: optional local aesthetic-predictor score, 0-1, None if
                      the aesthetic model isn't installed/enabled.
    technical_score: resolution/corruption/blank-image/duplicate checks, 0-1.
    overall_score:   weighted combination used against QUALITY_THRESHOLD.
    """

    semantic_score: float = Field(ge=0.0, le=1.0)
    aesthetic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    technical_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons for the score, consumed by the "
        "Prompt Optimizer on retry (e.g. 'subject occupies too little of "
        "frame', 'low sharpness detected').",
    )


class Candidate(BaseModel):
    """A single generated image candidate awaiting/after evaluation."""

    candidate_id: str
    file_path: str
    prompt: ImagePrompt
    attempt_number: int = Field(ge=1)
    evaluation: EvaluationResult | None = None


class GeneratedAsset(BaseModel):
    """The final, selected asset for one AssetPlan after the full
    generate -> evaluate -> (retry) -> select loop.
    """

    asset_id: str
    file_path: str
    prompt: ImagePrompt
    evaluation: EvaluationResult
    attempts_used: int = Field(ge=1)
    all_candidates: list[Candidate] = Field(default_factory=list)
