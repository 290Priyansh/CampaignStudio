"""
Phase 5 tests for evaluation/technical_checks.py.

These use real (tiny) images generated with PIL rather than mocks, since
the whole point of this module is deterministic pixel-level checks -- if
it's wrong, we want a real image to catch it, not a mock that assumes
away the behavior under test.
"""

from __future__ import annotations

from PIL import Image

from evaluation.technical_checks import (
    check_duplicate,
    check_file_validity,
    check_not_blank,
    check_resolution,
    run_technical_checks,
)


def _make_image(path, size=(512, 512), color=(120, 60, 200), noise=True):
    img = Image.new("RGB", size, color)
    if noise:
        px = img.load()
        for x in range(0, size[0], 7):
            for y in range(0, size[1], 11):
                px[x, y] = ((x * 3) % 255, (y * 5) % 255, (x + y) % 255)
    img.save(path)
    return path


def test_check_file_validity_missing_file(tmp_path):
    valid, feedback = check_file_validity(tmp_path / "nope.png")
    assert valid is False
    assert "missing or empty" in feedback[0]


def test_check_file_validity_corrupted_file(tmp_path):
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not actually a png")
    valid, feedback = check_file_validity(p)
    assert valid is False
    assert "not a valid" in feedback[0]


def test_check_file_validity_real_image(tmp_path):
    p = _make_image(tmp_path / "ok.png")
    valid, feedback = check_file_validity(p)
    assert valid is True
    assert feedback == []


def test_check_resolution_below_minimum(tmp_path):
    p = _make_image(tmp_path / "tiny.png", size=(64, 64))
    score, feedback = check_resolution(p)
    assert score == 0.0
    assert "below minimum" in feedback[0]


def test_check_resolution_mismatch_vs_expected(tmp_path):
    p = _make_image(tmp_path / "wrongsize.png", size=(512, 512))
    score, feedback = check_resolution(p, expected_width=1024, expected_height=1024)
    assert score == 0.7
    assert "does not match requested" in feedback[0]


def test_check_resolution_matches_expected(tmp_path):
    p = _make_image(tmp_path / "rightsize.png", size=(512, 512))
    score, feedback = check_resolution(p, expected_width=512, expected_height=512)
    assert score == 1.0
    assert feedback == []


def test_check_not_blank_flags_solid_color(tmp_path):
    p = _make_image(tmp_path / "blank.png", color=(200, 200, 200), noise=False)
    score, feedback = check_not_blank(p)
    assert score == 0.0
    assert "blank" in feedback[0]


def test_check_not_blank_passes_noisy_image(tmp_path):
    p = _make_image(tmp_path / "noisy.png")
    score, feedback = check_not_blank(p)
    assert score == 1.0
    assert feedback == []


def test_check_duplicate_detects_exact_repeat(tmp_path):
    p1 = _make_image(tmp_path / "a.png")
    p2 = tmp_path / "b.png"
    p2.write_bytes(p1.read_bytes())  # byte-identical copy

    seen: set[bytes] = set()
    score1, feedback1, hash1 = check_duplicate(p1, seen)
    seen.add(hash1)
    score2, feedback2, hash2 = check_duplicate(p2, seen)

    assert score1 == 1.0
    assert score2 == 0.0
    assert "duplicate" in feedback2[0]
    assert hash1 == hash2


def test_run_technical_checks_short_circuits_on_invalid_file(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"garbage")
    score, feedback, file_hash = run_technical_checks(p)
    assert score == 0.0
    assert file_hash is None
    assert any("not a valid" in f for f in feedback)


def test_run_technical_checks_averages_passing_image(tmp_path):
    p = _make_image(tmp_path / "good.png", size=(1024, 1024))
    score, feedback, file_hash = run_technical_checks(
        p, expected_width=1024, expected_height=1024, previous_hashes=set()
    )
    assert score == 1.0
    assert feedback == []
    assert file_hash is not None
