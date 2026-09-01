"""
Cheap, deterministic technical checks on a generated image -- no ML model
required. These catch "generation obviously failed" cases (corrupted file,
blank/solid-color output, wrong resolution, exact duplicate of a prior
candidate) before spending CLIP/aesthetic compute on an image that's
already known to be broken.
"""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

MIN_RESOLUTION = 256  # pixels, either dimension


def _file_hash(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).digest()


def check_file_validity(path: str | Path) -> tuple[bool, list[str]]:
    """Confirm the file exists, is non-empty, and PIL can decode it."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False, ["output file is missing or empty"]
    try:
        from PIL import Image

        with Image.open(p) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a hard fail here
        return False, [f"file is not a valid/decodable image: {exc}"]
    return True, []


def check_resolution(
    path: str | Path,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> tuple[float, list[str]]:
    from PIL import Image

    feedback: list[str] = []
    with Image.open(path) as img:
        w, h = img.size

    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        feedback.append(f"resolution {w}x{h} is below minimum {MIN_RESOLUTION}px")
        return 0.0, feedback

    if expected_width and expected_height and (w, h) != (expected_width, expected_height):
        feedback.append(
            f"resolution {w}x{h} does not match requested {expected_width}x{expected_height}"
        )
        return 0.7, feedback

    return 1.0, feedback


def check_not_blank(path: str | Path, std_threshold: float = 3.0) -> tuple[float, list[str]]:
    """Flag near-solid-color output -- a common diffusion failure mode
    (all-black/all-white/flat-gray from a broken pipeline call), detected
    via pixel standard deviation. No ML model needed for this check."""
    from PIL import Image

    with Image.open(path) as img:
        pixels = list(img.convert("L").getdata())

    stdev = statistics.pstdev(pixels) if len(pixels) > 1 else 0.0
    if stdev < std_threshold:
        return 0.0, [f"image appears blank/near-solid-color (pixel stdev={stdev:.2f})"]
    return 1.0, []


def hash_file(path: str | Path) -> bytes:
    """Public content-hash helper. Exposed so callers (e.g. the asset
    generation workflow) can accumulate hashes of every candidate they've
    generated so far -- check_duplicate() only compares, it doesn't mutate
    the caller's `previous_hashes` set itself."""
    return _file_hash(path)


def check_duplicate(path: str | Path, previous_hashes: set[bytes]) -> tuple[float, list[str], bytes]:
    """Compare against hashes of previously generated candidates in this run."""
    h = _file_hash(path)
    if h in previous_hashes:
        return 0.0, ["duplicate of a previously generated candidate"], h
    return 1.0, [], h


def run_technical_checks(
    path: str | Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    previous_hashes: set[bytes] | None = None,
) -> tuple[float, list[str], bytes | None]:
    """Run all technical checks and return (score in [0,1], feedback, file_hash).

    File validity is a hard gate (invalid file -> 0.0, early return, no
    point checking resolution of a file that can't be opened). Remaining
    checks are averaged.
    """
    valid, feedback = check_file_validity(path)
    if not valid:
        return 0.0, feedback, None

    scores: list[float] = []
    all_feedback: list[str] = []

    res_score, res_feedback = check_resolution(path, expected_width, expected_height)
    scores.append(res_score)
    all_feedback.extend(res_feedback)

    blank_score, blank_feedback = check_not_blank(path)
    scores.append(blank_score)
    all_feedback.extend(blank_feedback)

    file_hash = None
    if previous_hashes is not None:
        dup_score, dup_feedback, file_hash = check_duplicate(path, previous_hashes)
        scores.append(dup_score)
        all_feedback.extend(dup_feedback)

    overall = sum(scores) / len(scores)
    return overall, all_feedback, file_hash
