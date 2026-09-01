"""
Phase 4 tests for tools/image_generation.py.

No GPU, torch, or diffusers install is required: `torch` and `diffusers`
are injected as fake modules via sys.modules for the duration of these
tests, so we're testing this module's own control flow (path handling,
seeding, error mapping, backend selection) rather than real diffusion.
"""

from __future__ import annotations

import sys
import types

import pytest

from config.settings import Settings
from models.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tools.image_generation import (
    ComfyUIGenerator,
    DiffusersGenerator,
    get_image_generator,
)


@pytest.fixture
def fake_torch_and_diffusers(monkeypatch, tmp_path):
    """Install minimal fake `torch` and `diffusers` modules so
    DiffusersGenerator can run its real logic without either package
    actually being installed.
    """

    class _FakeImage:
        def save(self, path):
            # Write a tiny placeholder file so downstream path checks pass.
            with open(path, "wb") as f:
                f.write(b"FAKEPNG")

    class _FakeResult:
        def __init__(self):
            self.images = [_FakeImage()]

    class _FakePipe:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return _FakeResult()

    fake_pipeline_singleton = _FakePipe()

    class _FakeAutoPipeline:
        @staticmethod
        def from_pretrained(model_id, torch_dtype=None, use_auth_token=None):
            return fake_pipeline_singleton

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.AutoPipelineForText2Image = _FakeAutoPipeline

    class _FakeGenerator:
        def __init__(self, device):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Generator = _FakeGenerator

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    return fake_pipeline_singleton


def test_get_image_generator_returns_diffusers_by_default(monkeypatch):
    monkeypatch.setattr("tools.image_generation.settings", Settings(_env_file=None, image_backend="diffusers"))
    gen = get_image_generator()
    assert isinstance(gen, DiffusersGenerator)


def test_get_image_generator_returns_comfyui_when_configured(monkeypatch):
    fake_settings = Settings(
        _env_file=None, image_backend="comfyui", comfyui_workflow_path="/tmp/fake_workflow.json"
    )
    monkeypatch.setattr("tools.image_generation.settings", fake_settings)
    gen = get_image_generator()
    assert isinstance(gen, ComfyUIGenerator)


def test_comfyui_generator_requires_workflow_path(monkeypatch):
    fake_settings = Settings(_env_file=None, image_backend="comfyui", comfyui_workflow_path=None)
    monkeypatch.setattr("tools.image_generation.settings", fake_settings)
    with pytest.raises(ImageBackendUnavailableError):
        ComfyUIGenerator()


def test_diffusers_generator_raises_typed_error_when_deps_missing(monkeypatch):
    # Ensure torch/diffusers are NOT importable for this test.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)
    gen = DiffusersGenerator(model_id="fake/model", device="cpu")
    with pytest.raises(ImageBackendUnavailableError):
        gen.generate("a cat")


def test_diffusers_generator_generate_produces_file(fake_torch_and_diffusers, monkeypatch, tmp_path):
    fake_settings = Settings(_env_file=None, output_dir=str(tmp_path))
    monkeypatch.setattr("tools.image_generation.settings", fake_settings)

    gen = DiffusersGenerator(model_id="fake/model", device="cpu")
    out_path = tmp_path / "hero.png"
    result = gen.generate(
        "a running shoe on a forest trail",
        negative_prompt="blurry, watermark",
        width=512,
        height=512,
        seed=42,
        output_path=out_path,
    )

    assert result.file_path == str(out_path)
    assert result.seed == 42
    assert result.width == 512
    assert result.height == 512
    assert out_path.exists()
    # Pipeline should be cached on the instance after first call.
    assert gen._pipe is fake_torch_and_diffusers


def test_diffusers_generator_wraps_pipeline_call_errors(fake_torch_and_diffusers, monkeypatch, tmp_path):
    fake_settings = Settings(_env_file=None, output_dir=str(tmp_path))
    monkeypatch.setattr("tools.image_generation.settings", fake_settings)

    def _boom(self, **kwargs):
        raise RuntimeError("CUDA out of memory")

    # Dunder methods are looked up on the class, not the instance, so patch
    # the class's __call__ rather than assigning to the instance.
    monkeypatch.setattr(type(fake_torch_and_diffusers), "__call__", _boom)
    gen = DiffusersGenerator(model_id="fake/model", device="cpu")

    with pytest.raises(ImageGenerationError):
        gen.generate("a running shoe")
