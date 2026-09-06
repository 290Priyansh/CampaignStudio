"""
Shared test fixtures.

`fake_crewai` installs a minimal fake `crewai` module into sys.modules for
the duration of a test, so agents/creative_director.py,
agents/audience_analyst.py, and agents/campaign_strategist.py (all of which
do `from crewai import ...` lazily inside functions) can be exercised
without the real crewai package installed. Each of those modules only ever
builds one Task per Crew, so the fake Crew.kickoff() attaches
`module._pending_result` as every task's `output.pydantic`.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_crewai(monkeypatch):
    module = types.ModuleType("crewai")

    class _FakeTaskOutput:
        def __init__(self, pydantic=None):
            self.pydantic = pydantic
            if pydantic is not None and hasattr(pydantic, "model_dump_json"):
                self.raw = pydantic.model_dump_json()
            else:
                self.raw = str(pydantic) if pydantic is not None else ""

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTask:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.description = kwargs.get("description", "")
            self.output = _FakeTaskOutput(pydantic=None)

    class _FakeCrew:
        def __init__(self, agents=None, tasks=None, process=None, verbose=False):
            self.agents = agents
            self.tasks = tasks

        def kickoff(self):
            if module._raise_on_kickoff:
                raise RuntimeError(module._raise_on_kickoff)
            for t in self.tasks:
                t.output = _FakeTaskOutput(pydantic=module._pending_result)
            return None

    class _Process:
        sequential = "sequential"

    class _FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.Agent = _FakeAgent
    module.Task = _FakeTask
    module.Crew = _FakeCrew
    module.Process = _Process
    module.LLM = _FakeLLM
    module._pending_result = None
    module._raise_on_kickoff = None

    monkeypatch.setitem(sys.modules, "crewai", module)
    return module
