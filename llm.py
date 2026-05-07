"""LLM provider abstraction.

Supports Ollama (default, local) and Gemini.
Both expose a uniform `chat(messages, tools=None) -> Message` interface
where Message has `.content`, `.tool_calls`, and `.thinking` (optional).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass
class Message:
    role: str = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


class LLM:
    """Base class for LLM providers."""

    name: str = "base"

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> Message:
        raise NotImplementedError


class OllamaLLM(LLM):
    """Ollama local model with tool-calling support.

    Tools follow the OpenAI/JSON-schema style:
        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": {...}}}
    """

    name = "ollama"

    def __init__(self, model: str = "gpt-oss:20b", host: str | None = None):
        if ollama is None:
            raise RuntimeError("ollama package not installed; run pip install ollama")
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if tools:
            kwargs["tools"] = tools

        resp = self.client.chat(**kwargs)
        msg = resp.get("message", {}) if isinstance(resp, dict) else resp.message

        # ollama python client returns either dict or pydantic-ish object
        def _g(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        content = _g(msg, "content", "") or ""
        thinking = _g(msg, "thinking", None)
        raw_tcs = _g(msg, "tool_calls", []) or []

        tool_calls: list[ToolCall] = []
        for tc in raw_tcs:
            fn = _g(tc, "function", {})
            name = _g(fn, "name", "")
            args = _g(fn, "arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=name, arguments=args, id=_g(tc, "id")))

        return Message(content=content, tool_calls=tool_calls, thinking=thinking)


class GeminiLLM(LLM):
    """Google Gemini fallback. Tool-calling supported but kept simple here."""

    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-pro", api_key: str | None = None):
        if genai is None:
            raise RuntimeError("google-generativeai not installed")
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> Message:
        # Naive flatten — Gemini's tool format differs; we treat this as
        # text-only fallback. The agent loop should prefer Ollama when tools
        # are provided.
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            prompt_parts.append(f"[{role}]\n{m.get('content','')}")
        prompt = "\n\n".join(prompt_parts)
        resp = self._model.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        return Message(content=resp.text or "")


def get_llm(provider: str = "ollama", **kwargs) -> LLM:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaLLM(**kwargs)
    if provider == "gemini":
        return GeminiLLM(**kwargs)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Helpers for building tool schemas from python callables
# ---------------------------------------------------------------------------

def tool_schema(
    name: str, description: str, parameters: dict
) -> dict:
    """Build an OpenAI-style tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
