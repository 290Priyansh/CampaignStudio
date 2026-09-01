"""
Central logging setup.

Call `setup_logging()` once at process start (app/main.py, ui/streamlit_app.py,
or a test conftest) to get the "[ComponentName] message" format used
throughout this project, e.g.:

    [CreativeDirector] Creating campaign strategy
    [PromptOptimizer] Creating image prompt
    [ImageGenerator] Generating candidate 1/4
    [Evaluator] CLIP score: 0.81

Every module should log via `logging.getLogger(__name__)` (or a short alias
set at the top of the file) rather than printing directly, so log level and
format stay centrally controlled.
"""

from __future__ import annotations

import logging

from config.settings import settings


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=level or settings.log_level,
        format="[%(name)s] %(message)s",
    )
