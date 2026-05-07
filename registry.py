"""Strudel registry: ground-truth lookup for valid identifiers.

Used by the agent's tools to:
  * provide accurate function signatures/examples on lookup, and
  * validate generated code against known identifiers (catch hallucinations).
"""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "strudel_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def all_function_names() -> set[str]:
    reg = load_registry()
    names: set[str] = set(reg.get("functions", {}).keys())
    # include declared aliases
    for fn in reg.get("functions", {}).values():
        for a in fn.get("aliases", []) or []:
            names.add(a)
    # signals are usable as values too (sine, saw, rand, ...)
    names.update(reg.get("signals", {}).keys())
    names.update(reg.get("random_choice", {}).keys())
    return names


def all_samples() -> set[str]:
    reg = load_registry()
    return set(reg.get("common_samples", []))


def all_drum_machines() -> set[str]:
    reg = load_registry()
    return set(reg.get("drum_machines", []))


def all_synths() -> set[str]:
    reg = load_registry()
    return set(reg.get("synths", []))


def lookup_function(name: str) -> dict | None:
    reg = load_registry()
    funcs = reg.get("functions", {})
    if name in funcs:
        return {"name": name, **funcs[name]}
    # search aliases
    for canon, info in funcs.items():
        if name in (info.get("aliases") or []):
            return {"name": canon, "via_alias": name, **info}
    return None


def list_functions(category: str | None = None) -> list[dict]:
    reg = load_registry()
    out = []
    for name, info in reg.get("functions", {}).items():
        if category and info.get("category") != category:
            continue
        out.append({"name": name, "category": info.get("category"),
                    "signature": info.get("signature")})
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Identifiers that look like JS keywords / common values we should not flag.
JS_BUILTINS = {
    "const", "let", "var", "function", "return", "true", "false", "null",
    "undefined", "if", "else", "for", "while", "switch", "case", "break",
    "continue", "new", "this", "typeof", "instanceof", "in", "of", "do",
    "try", "catch", "finally", "throw", "Math", "Number", "String", "Array",
    "Object", "JSON", "console", "log", "Promise", "async", "await", "x",
    "y", "p", "v", "n", "i", "j", "k", "t", "fn",
}


_IDENT_AT_DOT = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENT_TOPLEVEL = re.compile(r"(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_STRING_LITERAL = re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")
_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_strings(code: str) -> str:
    return _STRING_LITERAL.sub('""', code)


def _strip_comments(code: str) -> str:
    """Remove // line comments and /* block comments */ from JS code."""
    code = _BLOCK_COMMENT.sub("", code)
    code = _LINE_COMMENT.sub("", code)
    return code


def extract_identifiers(code: str) -> dict[str, list[str]]:
    """Return {'methods': [...], 'top_level': [...]} found in code (deduped)."""
    # Strip comments BEFORE strings — comments may contain quotes that would
    # otherwise confuse the string-literal regex.
    cleaned = _strip_comments(code)
    cleaned = _strip_strings(cleaned)
    methods = sorted(set(_IDENT_AT_DOT.findall(cleaned)))
    top = sorted(set(_IDENT_TOPLEVEL.findall(cleaned)))
    # remove top-level identifiers that are also methods (already counted)
    return {"methods": methods, "top_level": top}


_NOTE_RE = re.compile(r"^[a-gA-G][#b]?[0-9]$")


def extract_sample_tokens(code: str) -> list[str]:
    """Pull sample tokens out of s("...") / sound("...") strings."""
    out: list[str] = []
    for m in re.finditer(r"\b(?:s|sound)\(\s*[\"']([^\"']+)[\"']\s*\)", code):
        body = m.group(1)
        for tok in re.split(r"[\s\[\]<>,*?!@/|()]+", body):
            tok = tok.strip()
            if tok and tok != "~" and not tok.isdigit():
                # strip ":N" sample-index suffix
                tok = tok.split(":")[0]
                if tok and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
                    out.append(tok)
    return out


def extract_note_misuse(code: str) -> list[str]:
    """Find s()/sound() calls whose body contains note-name tokens.

    Returns the offending strings.
    """
    bad: list[str] = []
    for m in re.finditer(r"\b(?:s|sound)\(\s*[\"']([^\"']+)[\"']\s*\)", code):
        body = m.group(1)
        toks = [t for t in re.split(r"[\s\[\]<>,*?!@/|()]+", body) if t and t != "~"]
        if toks and all(_NOTE_RE.match(t.split(":")[0]) for t in toks):
            bad.append(body)
    return bad


def fuzzy_suggest(name: str, pool: set[str], n: int = 3) -> list[str]:
    return difflib.get_close_matches(name, pool, n=n, cutoff=0.6)


def validate_code(code: str) -> dict:
    """Return a structured validation report for Strudel code.

    Report fields:
        ok: bool
        issues: [{kind, name, suggestions}]
        notes: [str]
    """
    funcs = all_function_names()
    samples = all_samples()
    drum_machines = all_drum_machines()
    synths = all_synths()

    idents = extract_identifiers(code)
    issues: list[dict] = []
    notes: list[str] = []

    for m in idents["methods"]:
        if m in funcs or m in JS_BUILTINS:
            continue
        # underscore-prefixed (visualizers etc.) are fine if registered
        sugg = fuzzy_suggest(m, funcs)
        issues.append({
            "kind": "unknown_method",
            "name": m,
            "suggestions": sugg,
            "hint": "Method called as `.{}(...)` is not a known Strudel function.".format(m),
        })

    for t in idents["top_level"]:
        if t in funcs or t in JS_BUILTINS:
            continue
        # ignore single-letter / arrow-fn artifacts handled above
        if len(t) <= 1:
            continue
        sugg = fuzzy_suggest(t, funcs)
        issues.append({
            "kind": "unknown_call",
            "name": t,
            "suggestions": sugg,
            "hint": "Top-level call `{}(...)` is not a known Strudel factory/function.".format(t),
        })

    # Sample tokens — only warn if it doesn't look like a valid sample.
    bank_pool = samples | drum_machines | synths
    # Skip sample-token reporting for s()/sound() bodies that are actually note
    # misuse (handled separately as `notes_in_sample_call` below) so we don't
    # double-report each note as both a sample miss AND a misuse.
    note_misuse_bodies = set(extract_note_misuse(code))
    note_misuse_tokens: set[str] = set()
    for body in note_misuse_bodies:
        for tok in re.split(r"[\s\[\]<>,*?!@/|()]+", body):
            tok = tok.strip().split(":")[0]
            if tok:
                note_misuse_tokens.add(tok)
    for tok in extract_sample_tokens(code):
        if tok in note_misuse_tokens:
            continue
        # accept if substring matches any known sample family or synth
        if tok in samples or tok in synths:
            continue
        # accept tokens that look like a known prefix (e.g. '808bd' is in samples)
        if any(tok.startswith(s) or s.startswith(tok) for s in samples):
            continue
        sugg = fuzzy_suggest(tok, bank_pool)
        issues.append({
            "kind": "unknown_sample",
            "name": tok,
            "suggestions": sugg,
            "hint": (
                "Sample `{}` is not in the built-in registry. It may exist in a "
                "custom bank, but if you intended a built-in, consider one of the "
                "suggestions.".format(tok)
            ),
        })

    if "reverb(" in code:
        notes.append("`reverb()` is not a Strudel function — use `.room(amount)`.")
    if "echo(" in code:
        notes.append("`echo()` is not a Strudel function — use `.delay()` + `.delaytime()` + `.delayfeedback()`.")

    # Detect s("c2 eb2 ...") which should be note("c2 eb2 ...")
    for body in extract_note_misuse(code):
        issues.append({
            "kind": "notes_in_sample_call",
            "name": body,
            "suggestions": [f'note("{body}")'],
            "hint": (
                f's()/sound() takes sample names, not note names. The string '
                f'"{body}" looks like notes — use note("{body}") instead, '
                f'optionally chained with .s("sawtooth") for a synth.'
            ),
        })

    # Semantic check: stack/cat/seq must receive Pattern objects, not raw strings.
    # Detect e.g. stack("bd*4", "hh*8") which is invalid; should be stack(s("bd*4"), s("hh*8")).
    for fname in ("stack", "cat", "seq"):
        for m in re.finditer(rf"\b{fname}\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code):
            args_blob = m.group(1)
            # Look for top-level string literal arguments (not inside another call)
            depth = 0
            i = 0
            current_arg_starts_with_string = True
            arg_buf: list[str] = []
            args_split: list[str] = []
            while i < len(args_blob):
                ch = args_blob[i]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                if ch == "," and depth == 0:
                    args_split.append("".join(arg_buf).strip())
                    arg_buf = []
                else:
                    arg_buf.append(ch)
                i += 1
            if arg_buf:
                args_split.append("".join(arg_buf).strip())
            for a in args_split:
                if not a:
                    continue
                if (a.startswith('"') and a.endswith('"')) or (
                    a.startswith("'") and a.endswith("'")
                ):
                    issues.append({
                        "kind": "raw_string_in_factory",
                        "name": fname,
                        "suggestions": [f"{fname}(s({a}), ...)", f"{fname}(note({a}), ...)"],
                        "hint": (
                            f"`{fname}(...)` takes Pattern objects, not raw mini-notation "
                            f"strings. Wrap {a} with s(), sound(), note() or n()."
                        ),
                    })

    return {"ok": len(issues) == 0, "issues": issues, "notes": notes}


if __name__ == "__main__":
    # quick self-test
    import sys
    sample_code = 's("bd sd hh cp").reverb(.3).filter(800).fart(2)'
    print(json.dumps(validate_code(sample_code), indent=2))
