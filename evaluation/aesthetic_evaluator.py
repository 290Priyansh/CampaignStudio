"""
Optional local aesthetic scorer.

A LAION-Aesthetics-style predictor is a small MLP trained on top of frozen
CLIP embeddings. This project does not bundle its weights (a separate
~5MB download -- see README 'Optional: aesthetic scoring'). If no weights
file is configured/present, `score()` returns None rather than fabricating
a number; `EvaluationResult.aesthetic_score` is `Optional[float]` for
exactly this reason (project brief section 14: "if practical").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.clip_evaluator import CLIPEvaluator

logger = logging.getLogger("Evaluator")


class AestheticEvaluator:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        clip_evaluator: "CLIPEvaluator | None" = None,
    ):
        self.weights_path = Path(weights_path) if weights_path else None
        self._clip_evaluator = clip_evaluator
        self._mlp = None

    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    def _load(self) -> None:
        if self._mlp is not None or not self.available():
            return
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("torch not installed; aesthetic scoring unavailable")
            return

        class _AestheticMLP(nn.Module):
            def __init__(self, input_dim: int = 512):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 1024), nn.ReLU(),
                    nn.Linear(1024, 128), nn.ReLU(),
                    nn.Linear(128, 16), nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

        mlp = _AestheticMLP()
        state = torch.load(self.weights_path, map_location="cpu")
        mlp.load_state_dict(state)
        mlp.eval()
        self._mlp = mlp

    def score(self, image_path: str) -> float | None:
        if not self.available():
            return None
        self._load()
        if self._mlp is None or self._clip_evaluator is None:
            return None

        import torch
        from PIL import Image

        self._clip_evaluator._load()
        image = self._clip_evaluator._preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0)

        with torch.no_grad():
            embedding = self._clip_evaluator._model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            raw_score = self._mlp(embedding.float()).item()

        # The LAION aesthetic predictor outputs roughly 0-10; normalize to 0-1.
        return max(0.0, min(1.0, raw_score / 10.0))
