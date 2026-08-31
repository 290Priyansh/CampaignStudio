"""
Structured image-generation prompt, produced by the Prompt Optimizer agent
and consumed directly by the ImageGenerator tool interface.

This is the core artifact of section 12 of the project brief: the system
never passes a raw campaign description straight to the diffusion model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImagePrompt(BaseModel):
    positive_prompt: str = Field(
        description="Full positive prompt: subject, environment, composition, "
        "lighting, camera, style, mood, color palette -- assembled into one "
        "diffusion-ready string."
    )
    negative_prompt: str = Field(
        description="Unwanted artifacts/qualities to exclude, comma-separated."
    )
    aspect_ratio: str = Field(description="e.g. '1:1', '4:5', '16:9'.")
    style: str
    lighting: str
    composition: str
    camera: str | None = Field(
        default=None, description="Camera/lens/perspective notes, if applicable."
    )
    subject: str = Field(description="Primary subject of the image, isolated for logging/eval.")

    # Generation parameters -- optional overrides of the config defaults,
    # left None to fall back to settings.image_* values.
    guidance_scale: float | None = None
    num_inference_steps: int | None = None

    def revised(self, feedback: list[str], revision_note: str) -> "ImagePrompt":
        """Return a copy of this prompt annotated with a revision note.

        The actual text rewriting is done by the Prompt Optimizer agent using
        `feedback` from the evaluator; this helper just keeps the audit trail
        (what changed and why) attached to the object for logging purposes.
        """
        data = self.model_dump()
        data["positive_prompt"] = f"{self.positive_prompt} | revision: {revision_note}"
        return ImagePrompt(**data)
