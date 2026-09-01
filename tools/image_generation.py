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
                use_auth_token=settings.hf_token,
            )
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

        width = width or settings.image_width
        height = height or settings.image_height
        steps = kwargs.get("num_inference_steps", settings.image_num_inference_steps)
        guidance = kwargs.get("guidance_scale", settings.image_guidance_scale)
        seed = seed if seed is not None else int(time.time())

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

        image = result.images[0]
        out_path = Path(output_path) if output_path else Path(settings.output_dir) / f"{uuid.uuid4().hex}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

        return GeneratedImageFile(file_path=str(out_path), seed=seed, width=width, height=height)


class ComfyUIGenerator(ImageGenerator):
    """Minimal ComfyUI HTTP client.

    Honest scope: this submits a prompt into an exported ComfyUI workflow
    (API-format JSON) by injecting text into any `CLIPTextEncode` node, and
    posts it to `/prompt`. It does NOT implement result polling via the
    `/history` endpoint, because that depends on the specific `SaveImage`
    node name/output path in *your* exported workflow -- a generic
    implementation would be either fragile or would need to reimplement a
    chunk of ComfyUI's own client. `generate()` raises NotImplementedError
    at that point with instructions, rather than silently returning a fake
    success. Diffusers is the supported default; treat this as a documented
    extension point, not a finished backend.
    """

    def __init__(self, base_url: str | None = None, workflow_path: str | None = None):
        self.base_url = base_url or settings.comfyui_url
        self.workflow_path = workflow_path or settings.comfyui_workflow_path
        if not self.workflow_path:
            raise ImageBackendUnavailableError(
                "COMFYUI_WORKFLOW_PATH is not set. Export a ComfyUI workflow as "
                "API-format JSON (Workflow > Export (API Format)) and point "
                "this setting at the resulting file."
            )

    def _check_server(self) -> None:
        import requests

        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ImageBackendUnavailableError(
                f"ComfyUI server unreachable at {self.base_url}: {exc}"
            ) from exc

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
        import json

        import requests

        self._check_server()

        workflow_template_path = Path(self.workflow_path)
        if not workflow_template_path.exists():
            raise ImageBackendUnavailableError(
                f"ComfyUI workflow file not found: {self.workflow_path}"
            )

        workflow = json.loads(workflow_template_path.read_text())
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                title = str(node.get("_meta", {}).get("title", "")).lower()
                node.setdefault("inputs", {})
                node["inputs"]["text"] = (negative_prompt or "") if "negative" in title else prompt

        try:
            resp = requests.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise ImageGenerationError(f"Failed to submit workflow to ComfyUI: {exc}") from exc

        raise NotImplementedError(
            "ComfyUI workflow submitted successfully (job queued), but result "
            "polling is workflow-specific and not implemented generically. "
            "Add a SaveImage-node-aware polling loop for your exported "
            "workflow, or use IMAGE_BACKEND=diffusers (the supported default)."
        )


def get_image_generator() -> ImageGenerator:
    """Factory reading `settings.image_backend`. The only place in the
    codebase that should branch on backend choice."""
    if settings.image_backend == "diffusers":
        return DiffusersGenerator()
    if settings.image_backend == "comfyui":
        return ComfyUIGenerator()
    raise ImageBackendUnavailableError(f"Unknown IMAGE_BACKEND: {settings.image_backend}")
