"""
Phase 5 tests for evaluation/quality_gate.py.

CLIP and aesthetic scoring are mocked here (they require real model
weights / a GPU for practical speed); technical checks run for real
against a tiny generated image, since they're cheap and deterministic.
"""

from __future__ import annotations

from PIL import Image

from evaluation import quality_gate
from schemas import ImagePrompt


def _make_prompt() -> ImagePrompt:
    return ImagePrompt(
        positive_prompt="a running shoe on a forest trail, golden hour",
        negative_prompt="blurry, watermark",
        aspect_ratio="4:5",
        style="editorial",
        lighting="golden hour",
        composition="rule of thirds",
        subject="running shoe",
    )


def _make_good_image(path):
    img = Image.new("RGB", (1024, 1024), (100, 150, 200))
    px = img.load()
    for x in range(0, 1024, 5):
        for y in range(0, 1024, 5):
            px[x, y] = ((x * 7) % 255, (y * 3) % 255, (x + y) % 255)
    img.save(path)
    return path


def test_evaluate_image_high_scores_pass_threshold(monkeypatch, tmp_path):
    img_path = _make_good_image(tmp_path / "good.png")

    monkeypatch.setattr(quality_gate._clip_evaluator, "score", lambda *a, **k: 0.9)
    monkeypatch.setattr(quality_gate._aesthetic_evaluator, "score", lambda *a, **k: None)

    result = quality_gate.evaluate_image(str(img_path), _make_prompt(), threshold=0.75)

    assert result.semantic_score == 0.9
    assert result.aesthetic_score is None
    assert result.technical_score == 1.0
    assert result.passed is True
    # semantic/technical only: (0.9*0.5 + 1.0*0.3) / 0.8 = 0.9375
    assert abs(result.overall_score - 0.9375) < 1e-3


def test_evaluate_image_low_semantic_score_fails_and_flags_feedback(monkeypatch, tmp_path):
    img_path = _make_good_image(tmp_path / "offtopic.png")

    monkeypatch.setattr(quality_gate._clip_evaluator, "score", lambda *a, **k: 0.2)
    monkeypatch.setattr(quality_gate._aesthetic_evaluator, "score", lambda *a, **k: None)

    result = quality_gate.evaluate_image(str(img_path), _make_prompt(), threshold=0.75)

    assert result.passed is False
    assert any("low semantic similarity" in f for f in result.feedback)
    assert any("below quality threshold" in f for f in result.feedback)


def test_evaluate_image_includes_aesthetic_when_available(monkeypatch, tmp_path):
    img_path = _make_good_image(tmp_path / "aesthetic.png")

    monkeypatch.setattr(quality_gate._clip_evaluator, "score", lambda *a, **k: 0.8)
    monkeypatch.setattr(quality_gate._aesthetic_evaluator, "score", lambda *a, **k: 0.6)

    result = quality_gate.evaluate_image(str(img_path), _make_prompt(), threshold=0.5)

    # 0.8*0.5 + 0.6*0.2 + 1.0*0.3 = 0.82
    assert abs(result.overall_score - 0.82) < 1e-3
    assert result.aesthetic_score == 0.6


def test_evaluate_image_invalid_file_scores_zero_technical(monkeypatch, tmp_path):
    bad_path = tmp_path / "bad.png"
    bad_path.write_bytes(b"not an image")

    monkeypatch.setattr(quality_gate._clip_evaluator, "score", lambda *a, **k: 0.0)
    monkeypatch.setattr(quality_gate._aesthetic_evaluator, "score", lambda *a, **k: None)

    result = quality_gate.evaluate_image(str(bad_path), _make_prompt(), threshold=0.75)
    assert result.technical_score == 0.0
    assert result.passed is False


def test_evaluate_image_handles_clip_scoring_exception_gracefully(monkeypatch, tmp_path):
    img_path = _make_good_image(tmp_path / "good2.png")

    def _boom(*a, **k):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(quality_gate._clip_evaluator, "score", _boom)
    monkeypatch.setattr(quality_gate._aesthetic_evaluator, "score", lambda *a, **k: None)

    result = quality_gate.evaluate_image(str(img_path), _make_prompt(), threshold=0.75)
    assert result.semantic_score == 0.0
    assert any("semantic scoring failed" in f for f in result.feedback)
    assert result.passed is False
