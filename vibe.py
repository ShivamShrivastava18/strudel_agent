"""Vibe catalog + vague-prompt detection.

The Strudel agent expects musical specifics (BPM, key, samples). Users often
give vague prompts like "make it sound like Daft Punk" or "rainy night". This
module turns those into concrete musical specs the agent can compose against.

Pipeline:
    detect: is_vague_prompt(text) -> bool
    lookup: interpret_vibe(text) -> {matches: [...], spec: {...}}
    fallback: web_search_vibe(name) via Tavily (if TAVILY_API_KEY set)
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

STYLES_PATH = Path(__file__).parent / "styles.json"


# Music-theory terms; if a prompt has many of these it's probably already specific.
MUSIC_TERMS = {
    "bpm", "tempo", "kick", "snare", "hi-hat", "hihat", "hh", "clap", "808",
    "909", "707", "cymbal", "ride", "rim", "shaker", "perc", "percussion",
    "bass", "lead", "pad", "arp", "arpeggio", "chord", "chords", "progression",
    "sawtooth", "saw", "square", "sine", "triangle", "fm", "synth", "synthesizer",
    "filter", "lpf", "hpf", "cutoff", "resonance", "reverb", "delay", "echo",
    "phaser", "chorus", "flanger", "tremolo", "vibrato", "distortion",
    "minor", "major", "scale", "key", "octave", "semitone", "interval",
    "rhythm", "polyrhythm", "swing", "groove", "16th", "8th", "quarter",
    "stack", "cat", "seq", "every", "fast", "slow", "rev",
    "loop", "bar", "beat", "measure", "time", "signature", "cycle",
}


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with open(STYLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_catalog() -> dict[str, dict]:
    """Flatten catalog into name -> entry, where entry includes 'type'."""
    cat = load_catalog()
    flat: dict[str, dict] = {}
    for top_key in ("artists", "genres", "moods", "scenes"):
        for name, entry in cat.get(top_key, {}).items():
            flat[name.lower()] = {**entry, "_type": top_key.rstrip("s"), "_name": name}
            for alias in entry.get("aliases", []) or []:
                flat[alias.lower()] = {**entry, "_type": top_key.rstrip("s"),
                                       "_name": name, "_via_alias": alias}
    return flat


def _resolve_based_on(entry: dict, flat: dict[str, dict]) -> dict:
    """If an entry has 'based_on', merge in fields from the parent (entry wins)."""
    base_name = entry.get("based_on")
    if not base_name:
        return entry
    base = flat.get(base_name.lower())
    if not base:
        return entry
    merged: dict = {**base, **entry}
    merged.pop("based_on", None)
    return merged


def all_known_names() -> list[str]:
    return sorted(_flatten_catalog().keys())


def is_vague_prompt(prompt: str) -> bool:
    """Heuristic: does this prompt need vibe interpretation?

    A prompt is "vague" when it mentions a known artist/genre/mood/scene, OR
    contains few musical-theory terms relative to its length.
    """
    p_lower = prompt.lower()
    flat = _flatten_catalog()
    # known reference present -> vague (we want the spec)
    for name in flat:
        if re.search(rf"\b{re.escape(name)}\b", p_lower):
            return True
    # very short and lacking music terms -> vague
    words = re.findall(r"[a-zA-Z]+", p_lower)
    if not words:
        return False
    music_hits = sum(1 for w in words if w in MUSIC_TERMS)
    if len(words) <= 8 and music_hits <= 1:
        return True
    return False


def interpret_vibe(prompt: str, k: int = 3) -> dict:
    """Find catalog matches for a prompt; return composite spec.

    Strategy:
      1. Direct substring matches against name + aliases (highest priority).
      2. Word-overlap scoring against entry feel/genres/text fields.
    Returns:
      {
        'prompt': str,
        'matches': [{type, name, why}, ...],
        'spec': {tempo_bpm, key_hint, drums, bass, melody, harmony, fx, feel,
                  palette_samples, palette_synths, drum_machine, reference_tracks}
      }
    """
    p_lower = prompt.lower()
    flat = _flatten_catalog()

    # 1. direct/alias matches
    matches: list[tuple[str, dict, str]] = []
    seen_names: set[str] = set()
    for name, entry in flat.items():
        if re.search(rf"\b{re.escape(name)}\b", p_lower):
            canon = entry["_name"]
            if canon in seen_names:
                continue
            seen_names.add(canon)
            via = entry.get("_via_alias")
            why = f"matched alias '{via}'" if via else f"matched name '{name}'"
            matches.append((canon, entry, why))

    # 2. fallback: keyword overlap on feel + genres
    if not matches:
        scored: list[tuple[float, str, dict, str]] = []
        words = set(re.findall(r"[a-zA-Z]+", p_lower))
        for name, entry in flat.items():
            text = " ".join([
                entry.get("feel", ""),
                " ".join(entry.get("genres", []) or []),
                entry.get("drums", ""),
                entry.get("melody", ""),
            ]).lower()
            entry_words = set(re.findall(r"[a-zA-Z]+", text))
            overlap = len(words & entry_words)
            if overlap >= 2:
                canon = entry["_name"]
                scored.append((overlap, canon, entry,
                               f"keyword overlap ({overlap})"))
        scored.sort(key=lambda x: -x[0])
        for _, canon, entry, why in scored[:k]:
            if canon not in seen_names:
                seen_names.add(canon)
                matches.append((canon, entry, why))

    matches = matches[:k]
    spec = _merge_specs([_resolve_based_on(e, flat) for _, e, _ in matches])

    return {
        "prompt": prompt,
        "matches": [{"type": e["_type"], "name": n, "why": w}
                    for n, e, w in matches],
        "spec": spec,
        "in_catalog": len(matches) > 0,
    }


def _merge_specs(entries: list[dict]) -> dict:
    """Combine multiple style entries into one spec.

    Lists are unioned; tempo ranges are intersected (or unioned if disjoint);
    text fields concatenate distinct values; first-wins for scalar hints.
    """
    if not entries:
        return {}
    spec: dict[str, Any] = {}

    def _add_list(key, vals):
        cur = spec.setdefault(key, [])
        for v in vals or []:
            if v and v not in cur:
                cur.append(v)

    text_fields = ("drums", "bass", "melody", "harmony", "fx", "feel")
    for f in text_fields:
        spec[f] = []

    tempo_lo: list[int] = []
    tempo_hi: list[int] = []

    for e in entries:
        for f in text_fields:
            v = e.get(f)
            if v and v not in spec[f]:
                spec[f].append(v)
        if "tempo_bpm" in e and isinstance(e["tempo_bpm"], list) and len(e["tempo_bpm"]) == 2:
            tempo_lo.append(e["tempo_bpm"][0])
            tempo_hi.append(e["tempo_bpm"][1])
        if "key_hint" in e and "key_hint" not in spec:
            spec["key_hint"] = e["key_hint"]
        if "drum_machine" in e and "drum_machine" not in spec:
            spec["drum_machine"] = e["drum_machine"]
        _add_list("palette_samples", e.get("palette_samples"))
        _add_list("palette_synths", e.get("palette_synths"))
        _add_list("genres", e.get("genres"))
        _add_list("reference_tracks", e.get("reference_tracks"))

    # Compress text-field lists into single strings
    for f in text_fields:
        spec[f] = "; ".join(spec[f]) if spec[f] else ""

    if tempo_lo and tempo_hi:
        spec["tempo_bpm"] = [max(tempo_lo), min(tempo_hi)] \
            if max(tempo_lo) <= min(tempo_hi) else [min(tempo_lo), max(tempo_hi)]
    return spec


# ---------------------------------------------------------------------------
# Tavily web-search fallback
# ---------------------------------------------------------------------------

def web_search_vibe(name: str) -> dict:
    """Use Tavily to research an unknown artist/style.

    Requires TAVILY_API_KEY in env. Returns a synthesized text summary the
    LLM can reason over.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "TAVILY_API_KEY not set in environment",
            "hint": (
                "Set TAVILY_API_KEY in .env to enable web search. Without it, "
                "the agent must rely on its own knowledge of the artist/style."
            ),
        }
    try:
        import requests
    except ImportError:
        return {"ok": False, "error": "requests package not available"}

    query = (
        f"What does the music of '{name}' sound like? Describe tempo (BPM), "
        f"key/mood, typical drums, bass, melody, instrumentation, and "
        f"production effects."
    )
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 5,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    answer = data.get("answer") or ""
    snippets = []
    for res in data.get("results", []) or []:
        snippets.append({
            "title": res.get("title"),
            "url": res.get("url"),
            "content": (res.get("content") or "")[:500],
        })
    return {"ok": True, "name": name, "answer": answer, "snippets": snippets}


# ---------------------------------------------------------------------------
# Pretty-printing for prompt injection
# ---------------------------------------------------------------------------

def render_spec_for_prompt(result: dict) -> str:
    """Format an interpret_vibe result as a compact spec block for the LLM."""
    if not result.get("matches"):
        return ""
    spec = result["spec"]
    matches_str = ", ".join(
        f"{m['name']} ({m['type']})" for m in result["matches"]
    )
    parts = [f"### Vibe spec (matched: {matches_str})"]
    if spec.get("tempo_bpm"):
        lo, hi = spec["tempo_bpm"]
        parts.append(f"- tempo: {lo}-{hi} BPM (use .cpm({(lo+hi)//2}/4))")
    if spec.get("key_hint"):
        parts.append(f"- key: {spec['key_hint']}")
    if spec.get("drum_machine"):
        parts.append(f"- drum machine: .bank(\"{spec['drum_machine']}\")")
    for f, label in (("drums", "drums"), ("bass", "bass"), ("melody", "melody"),
                     ("harmony", "harmony"), ("fx", "fx"), ("feel", "feel")):
        v = spec.get(f)
        if v:
            parts.append(f"- {label}: {v}")
    if spec.get("palette_samples"):
        parts.append(f"- preferred samples: {', '.join(spec['palette_samples'])}")
    if spec.get("palette_synths"):
        parts.append(f"- preferred synths: {', '.join(spec['palette_synths'])}")
    if spec.get("reference_tracks"):
        parts.append(f"- reference tracks: {', '.join(spec['reference_tracks'])}")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "make it sound like daft punk"
    print(f"is_vague_prompt: {is_vague_prompt(q)}")
    res = interpret_vibe(q)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n--- rendered ---")
    print(render_spec_for_prompt(res))
