"""
Local, open-source CLIP-based semantic evaluator.

Scores how well a generated image matches the prompt it was generated
from -- this is the concrete measurement that replaces the "ask the LLM if
this image looks good" anti-pattern the project brief explicitly rules out
(section 14).

Uses open_clip (ViT-B-32 / laion2b_s34b_b79k by default: ~600MB, works on
CPU though slower than GPU -- see README hardware section). Imports are
local to methods so this module can be imported without open-clip-torch
installed.
"""

from __future__ import annotations

import logging

from models.exceptions import ImageBackendUnavailableError

logger = logging.getLogger("Evaluator")


class CLIPEvaluator:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip  # noqa: F401
        except ImportError as exc:
            raise ImageBackendUnavailableError(
                "open-clip-torch is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        import open_clip

        logger.info("Loading CLIP model %s (%s) on %s", self.model_name, self.pretrained, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        model.eval()
        self._model, self._preprocess, self._tokenizer = model, preprocess, tokenizer

    def score(self, image_path: str, text: str) -> float:
        """Cosine similarity between image and text CLIP embeddings, rescaled
        from [-1, 1] to [0, 1] to stay consistent with the other scores this
        project combines it with."""
        self._load()
        import torch
        from PIL import Image

        image = self._preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        text_tokens = self._tokenizer([text]).to(self.device)

        with torch.no_grad():
            image_features = self._model.encode_image(image)
            text_features = self._model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()

        return max(0.0, min(1.0, (similarity + 1) / 2))
