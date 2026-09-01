"""
Tests for ui/streamlit_app.py.

Pure logic (StreamlitLogHandler, apply_ui_overrides, build_full_brief_text)
is unit-tested directly with no Streamlit runtime involved. The full app
is additionally smoke-tested with Streamlit's own `AppTest` harness
(streamlit.testing.v1), which actually executes the script in a simulated
session -- this catches import errors, widget-wiring mistakes, and crashes
on the "not yet run" path without needing a real browser or Ollama server.
"""

from __future__ import annotations

import logging

from config.settings import Settings
from ui.streamlit_app import apply_ui_overrides, build_full_brief_text, StreamlitLogHandler


def test_streamlit_log_handler_collects_formatted_records():
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger = logging.getLogger("TestComponent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("hello %s", "world")

    assert handler.records == ["[TestComponent] hello world"]
    logger.removeHandler(handler)


def test_apply_ui_overrides_mutates_settings_singleton(monkeypatch):
    import ui.streamlit_app as app_module

    fake_settings = Settings(_env_file=None)
    monkeypatch.setattr(app_module, "settings", fake_settings)

    apply_ui_overrides("comfyui", "gemma3:4b", 0.6)

    assert fake_settings.image_backend == "comfyui"
    assert fake_settings.ollama_model == "gemma3:4b"
    assert fake_settings.quality_threshold == 0.6


def test_build_full_brief_text_appends_optional_fields():
    text = build_full_brief_text("Launch an eco shoe", "College students", "energetic, modern")
    assert text.startswith("Launch an eco shoe")
    assert "Target audience: College students" in text
    assert "Brand style/personality: energetic, modern" in text


def test_build_full_brief_text_omits_empty_optional_fields():
    text = build_full_brief_text("Launch an eco shoe", "", "")
    assert text == "Launch an eco shoe"


def test_app_smoke_renders_initial_state_without_running_campaign():
    """Full AppTest run: the script should execute cleanly and show the
    'fill in a brief' prompt before the Generate button is clicked -- no
    network, Ollama, or GPU access should occur on this path."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("ui/streamlit_app.py")
    at.run()

    assert not at.exception
    assert any("Fill in a campaign brief" in info.value for info in at.info)


def test_app_smoke_errors_on_empty_brief_when_run_clicked():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("ui/streamlit_app.py")
    at.run()
    at.button[0].click().run()

    assert not at.exception
    assert any("cannot be empty" in err.value for err in at.error)
