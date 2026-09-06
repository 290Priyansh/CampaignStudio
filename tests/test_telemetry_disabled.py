"""
Confirms config.settings disables CrewAI's external telemetry call by
default -- see config/settings.py for why this matters for a project
whose core claim is fully local operation.
"""

from __future__ import annotations

import os

from config.settings import _disable_crewai_telemetry_by_default


def test_importing_config_disables_crewai_telemetry_by_default():
    # config.settings has already run at import time (by the time this test
    # module loads, other tests have imported it), so the env var should
    # already be set as a side effect of that import.
    import config.settings  # noqa: F401

    assert os.environ.get("OTEL_SDK_DISABLED") == "true"


def test_disable_function_does_not_override_explicit_value(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")

    _disable_crewai_telemetry_by_default()

    assert os.environ.get("OTEL_SDK_DISABLED") == "false"


def test_disable_function_sets_default_when_unset(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    _disable_crewai_telemetry_by_default()

    assert os.environ.get("OTEL_SDK_DISABLED") == "true"
