"""Tools exposed to the LLM agent.

Each tool is a Python callable plus a JSON-schema description. The agent
loop dispatches tool calls from the model into these functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import registry as reg
import vibe

ROOT = Path(__file__).parent
EXAMPLES_PATH = ROOT / "examples.jsonl"
INDEX_PATH = ROOT / "faiss_index.bin"
CHUNKS_PATH = ROOT / "doc_chunks.npy"
META_PATH = ROOT / "doc_metadata.npy"

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_embed_model: SentenceTransformer | None = None
_index = None
_chunks: np.ndarray | None = None
_meta: np.ndarray | None = None
_examples: list[dict] | None = None


def _ensure_loaded():
    global _embed_model, _index, _chunks, _meta, _examples
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    if _index is None and INDEX_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None and CHUNKS_PATH.exists():
        _chunks = np.load(CHUNKS_PATH, allow_pickle=True)
    if _meta is None and META_PATH.exists():
        _meta = np.load(META_PATH, allow_pickle=True)
    if _examples is None:
        _examples = []
        if EXAMPLES_PATH.exists():
            with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            _examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_docs(query: str, k: int = 5) -> dict:
    """Semantic search over Strudel documentation chunks."""
    _ensure_loaded()
    if _index is None or _chunks is None:
        return {"error": "FAISS index not built. Run embed_strudel_docs.py first."}
    q = _embed_model.encode([query], convert_to_numpy=True)
    D, I = _index.search(q, int(k))
    hits = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx < 0 or idx >= len(_chunks):
            continue
        hits.append({
            "rank": rank + 1,
            "score": float(-dist),  # negate distance so larger = better
            "source": str(_meta[idx]) if _meta is not None else "?",
            "text": str(_chunks[idx]),
        })
    return {"query": query, "hits": hits}


def lookup_function(name: str) -> dict:
    """Return signature + description + examples for a Strudel function."""
    info = reg.lookup_function(name)
    if info is None:
        suggestions = reg.fuzzy_suggest(name, reg.all_function_names())
        return {
            "found": False,
            "name": name,
            "suggestions": suggestions,
            "message": f"`{name}` is not a known Strudel function.",
        }
    return {"found": True, **info}


def list_functions(category: str | None = None) -> dict:
    """List functions, optionally filtered by category."""
    items = reg.list_functions(category)
    cats = sorted({i.get("category") for i in reg.list_functions()})
    return {"category_filter": category, "available_categories": cats,
            "count": len(items), "items": items}


def list_samples(kind: str = "samples") -> dict:
    """List built-in samples / drum machines / synths.

    kind: one of "samples", "drum_machines", "synths".
    """
    if kind == "samples":
        return {"kind": kind, "items": sorted(reg.all_samples())}
    if kind == "drum_machines":
        return {"kind": kind, "items": sorted(reg.all_drum_machines())}
    if kind == "synths":
        return {"kind": kind, "items": sorted(reg.all_synths())}
    return {"error": f"Unknown kind '{kind}'. Use samples|drum_machines|synths."}


def get_examples(query: str, k: int = 3) -> dict:
    """Retrieve curated working Strudel snippets matching the query.

    Uses simple tag/prompt keyword matching first, then falls back to
    semantic search over the prompt strings.
    """
    _ensure_loaded()
    examples = _examples or []
    if not examples:
        return {"hits": []}

    q_lower = query.lower()
    scored: list[tuple[float, dict]] = []
    for ex in examples:
        score = 0.0
        for t in ex.get("tags", []) or []:
            if t.lower() in q_lower:
                score += 2.0
        for word in q_lower.split():
            if word in ex.get("prompt", "").lower():
                score += 1.0
        if score > 0:
            scored.append((score, ex))

    if not scored:
        # semantic fallback over prompts
        prompts = [ex.get("prompt", "") for ex in examples]
        if _embed_model is not None and prompts:
            qv = _embed_model.encode([query], convert_to_numpy=True)
            pv = _embed_model.encode(prompts, convert_to_numpy=True)
            sims = (qv @ pv.T)[0]
            order = np.argsort(-sims)[:k]
            hits = [examples[i] for i in order]
            return {"query": query, "hits": hits}

    scored.sort(key=lambda x: -x[0])
    hits = [ex for _, ex in scored[:k]]
    return {"query": query, "hits": hits}


def validate_code(code: str) -> dict:
    """Validate a Strudel code snippet against the registry."""
    return reg.validate_code(code)


def interpret_vibe(prompt: str) -> dict:
    """Translate a vague reference (artist/genre/mood/scene) into a concrete
    musical spec (tempo, key, drums, bass, melody, fx, palette).
    """
    return vibe.interpret_vibe(prompt)


def web_search_vibe(name: str) -> dict:
    """Web-search (via Tavily) for an artist/style not in the local catalog.
    Requires TAVILY_API_KEY in the environment.
    """
    return vibe.web_search_vibe(name)


def final_answer(code: str, explanation: str = "") -> dict:
    """Marker tool: emits the final answer. Handled by the agent loop."""
    return {"code": code, "explanation": explanation}


# ---------------------------------------------------------------------------
# Tool registry & schemas
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "search_docs": search_docs,
    "lookup_function": lookup_function,
    "list_functions": list_functions,
    "list_samples": list_samples,
    "get_examples": get_examples,
    "validate_code": validate_code,
    "interpret_vibe": interpret_vibe,
    "web_search_vibe": web_search_vibe,
    "final_answer": final_answer,
}


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Semantic search over the Strudel documentation. Use this to "
                "discover features and find concrete usage examples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "k": {"type": "integer", "description": "Number of results.", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_function",
            "description": (
                "Look up a Strudel function by name. Returns the canonical signature, "
                "description and examples. Use to verify a function exists before using it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Function name e.g. 'lpf', 'jux'."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_functions",
            "description": (
                "List Strudel functions. Optionally filter by category "
                "(source, factory, time, conditional, stereo, amp, filter, envelope, fx, "
                "synth, tonal, math, viz, io, control, global)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_samples",
            "description": (
                "List built-in samples, drum-machine banks or synth waveforms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["samples", "drum_machines", "synths"],
                        "default": "samples",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_examples",
            "description": (
                "Fetch curated, hand-verified Strudel code snippets matching a query. "
                "Prefer using these as a template instead of writing from scratch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_code",
            "description": (
                "Validate a Strudel snippet. Reports unknown functions / samples and "
                "common pitfalls. ALWAYS call this on candidate code before final_answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interpret_vibe",
            "description": (
                "Translate a vague vibe reference (artist name, genre, mood, "
                "or scene like 'daft punk', 'lo-fi', 'sad', 'rainy night') "
                "into a concrete musical spec: tempo BPM range, key hint, "
                "drums, bass, melody, harmony, fx, palette of samples and "
                "synths. Call this FIRST when the user prompt is vague or "
                "references an artist/style/mood."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "User prompt or reference name."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_vibe",
            "description": (
                "Last-resort: search the web (via Tavily) for an artist/style "
                "not present in the local catalog. Returns a textual summary "
                "of the artist's typical tempo, instrumentation, and feel. "
                "Use ONLY after interpret_vibe returns no matches and you "
                "still need style context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Artist or style name."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Emit the final Strudel code and a short explanation. Only call this "
                "AFTER validate_code reports ok=true (or after fixing all issues)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The final Strudel code."},
                    "explanation": {"type": "string", "description": "Brief explanation."},
                },
                "required": ["code"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict) -> dict:
    """Run a tool by name with the given arguments. Returns a JSON-serializable dict."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}
