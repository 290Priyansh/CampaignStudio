"""
Image-generation backend abstraction.

Per section 13 of the brief: nothing outside this module should know or
care whether images come from local Diffusers or ComfyUI. Agents and the
workflow call `get_image_generator()` and use the returned object's
`generate()` method -- swapping IMAGE_BACKEND in `.env` is the only change
needed to switch backends.

    ImageGenerator (ABC)
        |
        ├── DiffusersGenerator   -- fully implemented, local HF Diffusers
        └── ComfyUIGenerator     -- minimal client; see class docstring for
                                     its honest scope/limits

torch/diffusers imports are local to methods (not module-level) so this
module -- and its control-flow (path handling, seeding, error mapping) --
can be imported and unit-tested without a GPU or those packages installed.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from models.exceptions import ImageBackendUnavailableError, ImageGenerationError

logger = logging.getLogger("ImageGenerator")


@dataclass
class GeneratedImageFile:
    file_path: str
    seed: int | None
    width: int
    height: int


class ImageGenerator(ABC):
    """Abstract interface every image-generation backend implements."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        output_path: str | Path | None = None,
        **kwargs,
    ) -> GeneratedImageFile:
        raise NotImplementedError


class DiffusersGenerator(ImageGenerator):
    """Local Hugging Face Diffusers backend.

    Lazily loads the pipeline on first `generate()` call and caches it on
    the instance -- loading an SDXL pipeline takes real time (and VRAM), so
    it should happen once per process, not once per image. See README for
    VRAM requirements at default settings (SDXL-base-1.0, fp16).
    """

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or settings.image_model
        self.device = device or settings.image_device
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ImageBackendUnavailableError(
                "diffusers/torch not installed. Run `pip install -r requirements.txt`. "
                "Note: practical SDXL generation speed requires a CUDA GPU -- see "
                "README hardware requirements for CPU-mode caveats."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(settings.image_dtype, torch.float16)

        logger.info(
            "Loading %s on %s (%s) -- this can take a while on first run",
            self.model_id, self.device, settings.image_dtype,
        )
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            if self.device == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe = pipe.to(self.device)
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            else:
                pipe = pipe.to(self.device)
        except Exception as exc:  # model download/OOM/incompatible-device errors
            raise ImageGenerationError(
                f"Failed to load diffusion pipeline '{self.model_id}' on "
                f"device '{self.device}': {exc}"
            ) from exc

        self._pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        output_path: str | Path | None = None,
        **kwargs,
    ) -> GeneratedImageFile:
        pipe = self._load_pipeline()  # validates torch/diffusers are importable first
        import torch  # safe now that _load_pipeline() above succeeded

        if self.device == "cuda":
            torch.cuda.empty_cache()

        width = int(width or settings.image_width or 512)
        height = int(height or settings.image_height or 512)
        
        steps_val = kwargs.get("num_inference_steps")
        steps = int(steps_val if steps_val is not None else (settings.image_num_inference_steps or 20))
        
        guidance_val = kwargs.get("guidance_scale")
        guidance = float(guidance_val if guidance_val is not None else (settings.image_guidance_scale or 7.5))
        
        seed = int(seed) if seed is not None else random.randint(0, 2147483647)

        generator = torch.Generator(self.device).manual_seed(seed)

        logger.info("Generating %dx%d image, seed=%d, steps=%d", width, height, seed, steps)
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        except Exception as exc:
            raise ImageGenerationError(f"Diffusion generation failed: {exc}") from exc
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

        image = result.images[0]
        out_path = Path(output_path) if output_path else Path(settings.output_dir) / f"{uuid.uuid4().hex}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

        return GeneratedImageFile(file_path=str(out_path), seed=seed, width=width, height=height)


def get_image_generator() -> ImageGenerator:
    """Factory returning the Diffusers image generator instance."""
    return DiffusersGenerator()
