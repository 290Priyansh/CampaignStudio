"""
Phase 7 tests for workflows/asset_generation.py.

The ImageGenerator, prompt_optimizer, and evaluate_asset boundaries are all
mocked -- this module's own job is orchestration logic (attempt counting,
ranking, revision triggering, hash accumulation), not real generation or
evaluation, which are covered by their own test files.
"""

from __future__ import annotations

from schemas import (
    AssetPlan,
    AssetType,
    Candidate,
    CreativeBrief,
    EvaluationResult,
    ImagePrompt,
)
from workflows import asset_generation


def _brief() -> CreativeBrief:
    return CreativeBrief(
        campaign_name="EcoStride Launch",
        campaign_objective="Drive awareness for a new eco-friendly running shoe.",
        target_audience="College students, 18-24",
        key_message="Move fast, tread light.",
        brand_personality=["energetic"],
        visual_style="bright editorial photography",
        color_direction=["forest green"],
        typography_direction="bold sans-serif",
        composition_guidelines="rule of thirds",
        photography_style="natural light",
        number_of_assets=1,
        recommended_aspect_ratios=["4:5"],
        content_ideas=["hero shot"],
        image_generation_strategy="product-in-context",
    )


def _asset() -> AssetPlan:
    return AssetPlan(
        asset_id="hero",
        asset_type=AssetType.HERO_IMAGE,
        purpose="Anchor visual",
        target_audience="College students",
        message="Statement of values",
        visual_direction="dramatic side-lit shot",
        aspect_ratio="4:5",
        prompt_requirements=["product in focus"],
    )


def _prompt(text="prompt v1") -> ImagePrompt:
    return ImagePrompt(
        positive_prompt=text, negative_prompt="blurry", aspect_ratio="4:5",
        style="editorial", lighting="golden hour", composition="rule of thirds",
        subject="running shoe",
    )


class _FakeGeneratedFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.seed = 1
        self.width = 896
        self.height = 1120


class _FakeImageGenerator:
    """Records every call and writes a tiny placeholder file per candidate."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.calls = 0

    def generate(self, prompt, negative_prompt=None, *, output_path=None, **kwargs):
        self.calls += 1
        with open(output_path, "wb") as f:
            f.write(f"fake-image-{self.calls}".encode())
        return _FakeGeneratedFile(str(output_path))


def _eval(score, passed, feedback=None):
    return EvaluationResult(
        semantic_score=score, technical_score=score, overall_score=score,
        passed=passed, feedback=feedback or [],
    )


def test_generate_asset_passes_on_first_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(asset_generation.settings, "candidates_per_asset", 2)
    monkeypatch.setattr(asset_generation.settings, "max_generation_attempts", 3)

    monkeypatch.setattr(asset_generation.prompt_optimizer, "create_initial_prompt", lambda b, a: _prompt())
    revise_calls = []
    monkeypatch.setattr(
        asset_generation.prompt_optimizer, "revise_prompt",
        lambda *a, **k: revise_calls.append(1) or _prompt("should not be called"),
    )

    call_results = [_eval(0.6, False), _eval(0.9, True)]

    def fake_evaluate(path, prompt, previous_hashes=None):
        return call_results.pop(0)

    monkeypatch.setattr(asset_generation, "evaluate_asset", fake_evaluate)

    gen = _FakeImageGenerator(tmp_path)
    result = asset_generation.generate_asset(_brief(), _asset(), gen, tmp_path)

    assert result.evaluation.passed is True
    assert result.evaluation.overall_score == 0.9
    assert result.attempts_used == 1
    assert gen.calls == 2  # candidates_per_asset=2, stopped after attempt 1
    assert revise_calls == []  # no revision needed


def test_generate_asset_revises_and_retries_then_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(asset_generation.settings, "candidates_per_asset", 1)
    monkeypatch.setattr(asset_generation.settings, "max_generation_attempts", 3)

    monkeypatch.setattr(asset_generation.prompt_optimizer, "create_initial_prompt", lambda b, a: _prompt("v1"))

    revise_calls = []

    def fake_revise(previous, feedback, brief, asset):
        revise_calls.append((previous.positive_prompt, list(feedback)))
        return _prompt(f"v{len(revise_calls) + 1}")

    monkeypatch.setattr(asset_generation.prompt_optimizer, "revise_prompt", fake_revise)

    # Attempt 1 fails, attempt 2 passes.
    results = [
        _eval(0.5, False, feedback=["subject too small"]),
        _eval(0.85, True),
    ]

    def fake_evaluate(path, prompt, previous_hashes=None):
        return results.pop(0)

    monkeypatch.setattr(asset_generation, "evaluate_asset", fake_evaluate)

    gen = _FakeImageGenerator(tmp_path)
    result = asset_generation.generate_asset(_brief(), _asset(), gen, tmp_path)

    assert result.evaluation.passed is True
    assert result.attempts_used == 2
    assert len(revise_calls) == 1
    assert revise_calls[0][0] == "v1"
    assert revise_calls[0][1] == ["subject too small"]
    assert len(result.all_candidates) == 2


def test_generate_asset_exhausts_attempts_and_returns_best_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(asset_generation.settings, "candidates_per_asset", 1)
    monkeypatch.setattr(asset_generation.settings, "max_generation_attempts", 2)

    monkeypatch.setattr(asset_generation.prompt_optimizer, "create_initial_prompt", lambda b, a: _prompt("v1"))
    monkeypatch.setattr(asset_generation.prompt_optimizer, "revise_prompt", lambda *a, **k: _prompt("v2"))

    results = [_eval(0.4, False, feedback=["bad"]), _eval(0.55, False, feedback=["still bad"])]

    def fake_evaluate(path, prompt, previous_hashes=None):
        return results.pop(0)

    monkeypatch.setattr(asset_generation, "evaluate_asset", fake_evaluate)

    gen = _FakeImageGenerator(tmp_path)
    result = asset_generation.generate_asset(_brief(), _asset(), gen, tmp_path)

    # Never passed, but returns the best of the two (0.55 > 0.4), and never
    # exceeds max_generation_attempts.
    assert result.evaluation.passed is False
    assert result.evaluation.overall_score == 0.55
    assert result.attempts_used == 2
    assert gen.calls == 2


def test_rank_candidates_sorts_best_first():
    prompt = _prompt()
    candidates = [
        Candidate(candidate_id="c1", file_path="a.png", prompt=prompt, attempt_number=1, evaluation=_eval(0.3, False)),
        Candidate(candidate_id="c2", file_path="b.png", prompt=prompt, attempt_number=1, evaluation=_eval(0.9, True)),
        Candidate(candidate_id="c3", file_path="c.png", prompt=prompt, attempt_number=1, evaluation=_eval(0.6, False)),
    ]
    ranked = asset_generation.rank_candidates(candidates)
    assert [c.candidate_id for c in ranked] == ["c2", "c3", "c1"]


def test_generate_candidates_accumulates_hashes_across_calls(monkeypatch, tmp_path):
    """The hash-accumulation fix: previous_hashes should grow after each
    candidate so later candidates are actually checked for duplicates."""
    monkeypatch.setattr(asset_generation.settings, "candidates_per_asset", 3)

    monkeypatch.setattr(asset_generation, "evaluate_asset", lambda path, prompt, previous_hashes=None: _eval(0.5, False))

    gen = _FakeImageGenerator(tmp_path)
    previous_hashes: set[bytes] = set()

    asset_generation.generate_candidates(gen, _prompt(), _asset(), 1, tmp_path, previous_hashes)

    # Each of the 3 candidates wrote distinct fake content, so 3 distinct hashes.
    assert len(previous_hashes) == 3
