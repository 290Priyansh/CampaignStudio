"""
Phase 3 tests for models/llm.py. No real Ollama server or network access
required -- every external call is mocked.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from models import llm
from models.exceptions import OllamaUnavailableError, StructuredOutputError


class _DummySchema(BaseModel):
    value: str


def test_get_langchain_llm_raises_typed_error_when_package_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_ollama":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OllamaUnavailableError):
        llm.get_langchain_llm()


def test_get_crewai_llm_raises_typed_error_when_package_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "crewai":
            raise ImportError("simulated missing package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OllamaUnavailableError):
        llm.get_crewai_llm()


def test_check_ollama_available_false_on_connection_error(monkeypatch):
    import requests

    class _FakeSession:
        def get(self, *args, **kwargs):
            raise requests.exceptions.ConnectionError("no server")

    monkeypatch.setattr(requests, "get", _FakeSession().get)
    assert llm.check_ollama_available() is False


def test_check_ollama_available_true_on_200(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse())
    assert llm.check_ollama_available() is True


def test_invoke_structured_returns_schema_instance_on_first_try(monkeypatch):
    class _FakeStructuredLLM:
        def invoke(self, prompt):
            return _DummySchema(value="ok")

    monkeypatch.setattr(llm, "get_structured_llm", lambda schema, temperature=None: _FakeStructuredLLM())

    result = llm.invoke_structured(_DummySchema, "some prompt")
    assert isinstance(result, _DummySchema)
    assert result.value == "ok"


def test_invoke_structured_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _FlakyStructuredLLM:
        def invoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ValueError("malformed output")
            return _DummySchema(value="recovered")

    monkeypatch.setattr(llm, "get_structured_llm", lambda schema, temperature=None: _FlakyStructuredLLM())

    result = llm.invoke_structured(_DummySchema, "some prompt", retries=2)
    assert result.value == "recovered"
    assert calls["n"] == 2


def test_invoke_structured_raises_after_exhausting_retries(monkeypatch):
    class _AlwaysFailsLLM:
        def invoke(self, prompt):
            raise ValueError("always malformed")

    monkeypatch.setattr(llm, "get_structured_llm", lambda schema, temperature=None: _AlwaysFailsLLM())

    with pytest.raises(StructuredOutputError):
        llm.invoke_structured(_DummySchema, "some prompt", retries=1)


def test_invoke_structured_coerces_dict_like_result(monkeypatch):
    class _DictReturningLLM:
        def invoke(self, prompt):
            return {"value": "coerced"}

    monkeypatch.setattr(llm, "get_structured_llm", lambda schema, temperature=None: _DictReturningLLM())

    result = llm.invoke_structured(_DummySchema, "some prompt")
    assert result.value == "coerced"
