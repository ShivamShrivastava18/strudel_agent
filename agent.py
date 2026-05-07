"""Strudel agent: tool-calling LLM that generates validated Strudel code.

Usage:
    python agent.py                       # interactive REPL with default Ollama model
    python agent.py "kick on every beat"  # one-shot prompt
    python agent.py --provider gemini     # use Gemini instead
    python agent.py --model gpt-oss:20b   # override Ollama model
    python agent.py --max-iters 10 --verbose

The agent uses tools (search_docs, lookup_function, list_samples,
get_examples, validate_code, final_answer) to ground generation in
verified Strudel facts. Every candidate snippet must pass validation
before being returned.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from dotenv import load_dotenv

import tools as toolset
import vibe
from llm import LLM, Message, get_llm

load_dotenv()


SYSTEM_PROMPT = """You are an expert Strudel live-coding agent. Strudel is a
JavaScript port of TidalCycles for making music with code. You must produce
runnable Strudel code that uses ONLY real Strudel functions and samples.

## Hard rules
1. NEVER invent function names. If unsure, call `lookup_function` or
   `search_docs` first.
2. ALWAYS call `validate_code` on your candidate snippet before
   `final_answer`. If validation reports issues, FIX the code (using
   `lookup_function` / `search_docs` / `get_examples` to find correct
   alternatives) and re-validate. Repeat until `ok=true`.
3. Prefer `get_examples` early to find a hand-verified template close to
   the user's request, then adapt it.
4. Output is plain Strudel/JavaScript - use method chaining like
   `s("bd*4").gain(.8)`, mini-notation in strings like `"bd ~ sd ~"`,
   and factories `stack(...)`, `cat(...)`, `seq(...)`, `chord(...)`.
5. Filter cutoffs are in Hz (e.g. 200..8000). Reverb is `.room(amount)`,
   NOT `reverb()`. Delay uses `.delay()` + `.delaytime()` + `.delayfeedback()`.
6. NOTES vs SAMPLES — never put note names in `s()` / `sound()`. Use
   `note("c2 eb2 g2").s("sawtooth")` for melodies/basslines. `s("c2")`
   is WRONG (c2 is a note, not a sample).
7. SINGLE TOP-LEVEL EXPRESSION — Strudel only plays the LAST top-level
   expression in a snippet. Multi-part patterns MUST be wrapped in ONE
   `stack(...)`. Do not emit several `s(...)`/`note(...)` statements
   side-by-side; combine them as `stack(part1, part2, part3)`.
8. NO COMMENTS in the final code — keep it clean. Save commentary for
   the `explanation` field of `final_answer`.

## Process (be efficient — aim for 3-5 tool calls total)
- If the user message contains a "### Vibe spec" block, USE IT as the brief —
  match its tempo (via `.cpm`), drum machine (via `.bank`), palette samples,
  and feel. Do NOT call `interpret_vibe` again.
- If there is no vibe spec and the prompt references an artist/genre/mood
  ("daft punk", "lo-fi", "sad", "rainy night"), call `interpret_vibe` first.
  If that returns no matches and you still need style info, fall back to
  `web_search_vibe`.
- Then try `get_examples(query)` — if a curated snippet matches, adapt it.
- Use `lookup_function` (not `search_docs`) when you only need to verify a
  single function name. Search docs only for genuinely novel concepts.
- After at most 2 search/lookup calls, commit to writing code.
- Call `validate_code(code)` once. Fix issues if any. Validate again.
- Call `final_answer(code, explanation)` as soon as validation passes.
- Stop searching once you have enough info — do not over-explore.

## Anti-drift rules (CRITICAL)
- DO NOT keep searching for a sample that isn't in Strudel. If the vibe
  spec lists `palette_samples`, USE THOSE. If you can't find an exotic
  sample (sitar, tabla, etc.) after ONE search, IMMEDIATELY substitute
  with `note(...).s("sine"|"triangle"|"sawtooth")` — synth lines can
  imply any instrument vibe.
- DO NOT call the same tool with the same arguments twice. If you've
  already searched for X, you have your answer. Move on.
- DO NOT re-call `interpret_vibe` if a "### Vibe spec" block is already
  in the user message — the spec is your source of truth.
- The user prompt is whatever appears under "User request:" in the most
  recent user message. DO NOT invent or substitute a different prompt.

Be terse. The user wants working code, not commentary.
"""


_FENCE = re.compile(r"```(?:javascript|js|strudel)?\n?(.*?)```", re.DOTALL)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "..."


def _extract_code_block(text: str) -> str:
    if not text:
        return ""
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


_STRUDEL_HINTS = re.compile(
    r"\b(?:s|sound|note|n|stack|cat|seq|chord|setcps|cpm|samples)\s*\("
)


def _looks_like_strudel(text: str) -> bool:
    """Heuristic: does this string contain at least one Strudel-style call?"""
    if not text:
        return False
    return bool(_STRUDEL_HINTS.search(text))


def _try_parse_final(text: str) -> dict | None:
    """If the model emitted a JSON dict with 'code' inline, parse it."""
    if not text:
        return None
    s = text.strip()
    # Strip optional fenced code block wrapping JSON
    fence_match = _FENCE.search(s)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        candidate = s
    # Find the first {...} block
    if "{" in candidate and "}" in candidate:
        start = candidate.find("{")
        end = candidate.rfind("}")
        try:
            obj = json.loads(candidate[start : end + 1])
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                return {"code": obj["code"],
                        "explanation": obj.get("explanation", "")}
        except json.JSONDecodeError:
            pass
    return None


def _format_tool_result(name: str, result: Any) -> str:
    """Compact JSON for tool results so we don't blow up context."""
    try:
        if isinstance(result, dict) and "hits" in result:
            hits = result.get("hits", [])
            for h in hits:
                if isinstance(h, dict) and "text" in h and isinstance(h["text"], str):
                    h["text"] = _truncate(h["text"], 600)
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


# Tools that count against the search budget — chatty/expensive ones the model
# tends to over-call. Lookup/list/validate stay free.
_SEARCH_TOOLS = {"search_docs", "web_search_vibe", "interpret_vibe"}
_SEARCH_BUDGET = 3


def _call_signature(name: str, args: dict) -> str:
    try:
        return f"{name}::{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f"{name}::{args!r}"


def _loop_guard_message(
    tc_name: str,
    tc_args: dict,
    seen_calls: dict[str, int],
    search_count: int,
) -> str | None:
    """If a tool call should be blocked, return a nudge to inject. Else None."""
    sig = _call_signature(tc_name, tc_args)
    if seen_calls.get(sig, 0) >= 1:
        return (
            f"You already called `{tc_name}` with the same arguments. "
            "Do not repeat the same lookup. Use what you already have and "
            "commit to writing code now (call `validate_code`, then `final_answer`)."
        )
    if tc_name in _SEARCH_TOOLS and search_count >= _SEARCH_BUDGET:
        return (
            f"You have already used {search_count} search calls. STOP searching "
            "and commit to code. Use the palette samples from the vibe spec, or "
            "substitute exotic samples with `note(...).s(\"sine\"|\"triangle\"|\"sawtooth\")`. "
            "Call `validate_code(code)`, then `final_answer(code, explanation)`."
        )
    return None


def run_agent(
    user_prompt: str,
    llm: LLM,
    max_iters: int = 12,
    verbose: bool = False,
) -> dict:
    """Run the tool-calling loop and return {'code', 'explanation', 'history'}."""
    # Auto-detect vague prompts (artist/genre/mood/scene references) and
    # prepend a concrete vibe spec so the model doesn't have to guess.
    enriched_prompt = user_prompt
    vibe_info: dict | None = None
    if vibe.is_vague_prompt(user_prompt):
        vibe_info = vibe.interpret_vibe(user_prompt)
        spec_block = vibe.render_spec_for_prompt(vibe_info)
        if spec_block:
            enriched_prompt = (
                f"{spec_block}\n\n"
                f"User request: {user_prompt}\n\n"
                "Use the spec above as a starting point. You may still call "
                "`interpret_vibe` or `web_search_vibe` for more detail, but "
                "don't repeat the lookup that was already done."
            )
            if verbose:
                print(
                    f"[vibe] auto-injected spec for: "
                    f"{[m['name'] for m in vibe_info.get('matches', [])]}",
                    file=sys.stderr,
                )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": enriched_prompt},
    ]
    history: list[dict] = []
    if vibe_info is not None:
        history.append({"step": -1, "auto_vibe": vibe_info})
    final: dict | None = None
    seen_calls: dict[str, int] = {}
    search_count = 0

    for step in range(max_iters):
        msg: Message = llm.chat(
            messages, tools=toolset.TOOL_SCHEMAS, temperature=0.2
        )

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        # ollama's pydantic Message wants arguments as a dict
                        "arguments": tc.arguments if isinstance(tc.arguments, dict) else {},
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if verbose:
            if msg.thinking:
                print(
                    f"\n[step {step}] thinking: {_truncate(msg.thinking, 200)}",
                    file=sys.stderr,
                )
            if msg.content:
                print(
                    f"[step {step}] assistant: {_truncate(msg.content, 200)}",
                    file=sys.stderr,
                )
            for tc in msg.tool_calls:
                print(
                    f"[step {step}] tool_call: {tc.name}({_truncate(json.dumps(tc.arguments), 120)})",
                    file=sys.stderr,
                )

        if not msg.tool_calls:
            history.append({"step": step, "assistant": msg.content})
            # Fallback: model may have emitted a JSON object that mimics
            # final_answer in the content. Try to parse it.
            parsed = _try_parse_final(msg.content)
            if parsed is not None:
                check = toolset.validate_code(parsed.get("code", ""))
                if check["ok"]:
                    return {**parsed, "history": history}
                # Re-prompt the model to fix it, instead of giving up.
                messages.append({
                    "role": "user",
                    "content": (
                        "Your inline answer failed validation:\n"
                        + json.dumps(check, ensure_ascii=False)
                        + "\nCall the `final_answer` tool with corrected code."
                    ),
                })
                continue
            extracted = _extract_code_block(msg.content)
            if not _looks_like_strudel(extracted):
                nudge = (
                    "Your reply did not contain any Strudel code or tool calls. "
                    "Stop reasoning in prose. Call `get_examples` for a starter, "
                    "OR write Strudel code now and call `validate_code` then "
                    "`final_answer`. Use the vibe spec's palette samples directly."
                )
                history.append({"step": step, "guard": nudge})
                messages.append({"role": "user", "content": nudge})
                if verbose:
                    print(f"[step {step}] guard: {nudge}", file=sys.stderr)
                continue
            return {
                "code": extracted,
                "explanation": msg.content,
                "history": history,
            }

        # Loop guard — block repeated identical calls and over-budget searches.
        guard_msg: str | None = None
        for tc in msg.tool_calls:
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            guard_msg = _loop_guard_message(tc.name, args, seen_calls, search_count)
            if guard_msg:
                break
        if guard_msg:
            history.append({"step": step, "guard": guard_msg})
            messages.append({"role": "user", "content": guard_msg})
            if verbose:
                print(f"[step {step}] guard: {guard_msg}", file=sys.stderr)
            continue

        for i, tc in enumerate(msg.tool_calls):
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            sig = _call_signature(tc.name, args)
            seen_calls[sig] = seen_calls.get(sig, 0) + 1
            if tc.name in _SEARCH_TOOLS:
                search_count += 1
            result = toolset.dispatch(tc.name, tc.arguments)
            content = _format_tool_result(tc.name, result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id or f"call_{step}_{i}",
                "name": tc.name,
                "content": content,
            })
            history.append({
                "step": step,
                "tool": tc.name,
                "args": tc.arguments,
                "result": result,
            })
            if verbose:
                print(
                    f"[step {step}] -> {tc.name} result: {_truncate(content, 200)}",
                    file=sys.stderr,
                )
            if tc.name == "final_answer":
                final = result

        if final is not None:
            check = toolset.validate_code(final.get("code", ""))
            if not check["ok"]:
                final = None
                messages.append({
                    "role": "user",
                    "content": (
                        "validate_code on your final_answer reported issues:\n"
                        + json.dumps(check, ensure_ascii=False)
                        + "\nFix the code and call final_answer again."
                    ),
                })
                continue
            return {**final, "history": history}

    return {
        "code": "",
        "explanation": "Agent did not converge within max_iters.",
        "history": history,
    }


def run_agent_stream(
    user_prompt: str,
    llm: LLM,
    max_iters: int = 12,
):
    """Generator version of run_agent that yields event dicts.

    Event types:
      {"type": "vibe", "matches": [...], "spec": {...}}
      {"type": "thinking", "step": int, "content": str}
      {"type": "assistant", "step": int, "content": str}
      {"type": "tool_call", "step": int, "name": str, "arguments": dict}
      {"type": "tool_result", "step": int, "name": str, "result": Any}
      {"type": "final", "code": str, "explanation": str}
      {"type": "error", "message": str}
      {"type": "done"}
    """
    enriched_prompt = user_prompt
    if vibe.is_vague_prompt(user_prompt):
        vibe_info = vibe.interpret_vibe(user_prompt)
        spec_block = vibe.render_spec_for_prompt(vibe_info)
        if spec_block:
            enriched_prompt = (
                f"{spec_block}\n\n"
                f"User request: {user_prompt}\n\n"
                "Use the spec above as a starting point. You may still call "
                "`interpret_vibe` or `web_search_vibe` for more detail, but "
                "don't repeat the lookup that was already done."
            )
            yield {
                "type": "vibe",
                "matches": vibe_info.get("matches", []),
                "spec": vibe_info.get("spec", {}),
            }

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": enriched_prompt},
    ]
    final: dict | None = None
    seen_calls: dict[str, int] = {}
    search_count = 0

    for step in range(max_iters):
        try:
            msg: Message = llm.chat(
                messages, tools=toolset.TOOL_SCHEMAS, temperature=0.2
            )
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield {"type": "done"}
            return

        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, dict) else {},
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if msg.thinking:
            yield {"type": "thinking", "step": step, "content": msg.thinking}
        if msg.content:
            yield {"type": "assistant", "step": step, "content": msg.content}

        if not msg.tool_calls:
            parsed = _try_parse_final(msg.content)
            if parsed is not None:
                check = toolset.validate_code(parsed.get("code", ""))
                if check["ok"]:
                    yield {
                        "type": "final",
                        "code": parsed["code"],
                        "explanation": parsed.get("explanation", ""),
                    }
                    yield {"type": "done"}
                    return
                messages.append({
                    "role": "user",
                    "content": (
                        "Your inline answer failed validation:\n"
                        + json.dumps(check, ensure_ascii=False)
                        + "\nCall the `final_answer` tool with corrected code."
                    ),
                })
                continue
            # No tool calls, no JSON. If the content doesn't even look like
            # Strudel code, the model rambled — re-prompt it with a hard
            # directive instead of emitting garbage as the final answer.
            extracted = _extract_code_block(msg.content)
            if not _looks_like_strudel(extracted):
                nudge = (
                    "Your reply did not contain any Strudel code or tool calls. "
                    "Stop reasoning in prose. Call `get_examples` for a starter, "
                    "OR write Strudel code now and call `validate_code` then "
                    "`final_answer`. Use the vibe spec's palette samples directly."
                )
                yield {"type": "guard", "step": step, "message": nudge}
                messages.append({"role": "user", "content": nudge})
                continue
            yield {
                "type": "final",
                "code": extracted,
                "explanation": msg.content or "",
            }
            yield {"type": "done"}
            return

        # Loop guard — block repeated identical calls and over-budget searches.
        guard_msg: str | None = None
        for tc in msg.tool_calls:
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            guard_msg = _loop_guard_message(tc.name, args, seen_calls, search_count)
            if guard_msg:
                break
        if guard_msg:
            yield {"type": "guard", "step": step, "message": guard_msg}
            messages.append({"role": "user", "content": guard_msg})
            continue

        for i, tc in enumerate(msg.tool_calls):
            args = tc.arguments if isinstance(tc.arguments, dict) else {}
            seen_calls[_call_signature(tc.name, args)] = (
                seen_calls.get(_call_signature(tc.name, args), 0) + 1
            )
            if tc.name in _SEARCH_TOOLS:
                search_count += 1
            yield {
                "type": "tool_call",
                "step": step,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            result = toolset.dispatch(tc.name, tc.arguments)
            content = _format_tool_result(tc.name, result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id or f"call_{step}_{i}",
                "name": tc.name,
                "content": content,
            })
            yield {
                "type": "tool_result",
                "step": step,
                "name": tc.name,
                "result": result,
            }
            if tc.name == "final_answer":
                final = result

        if final is not None:
            check = toolset.validate_code(final.get("code", ""))
            if not check["ok"]:
                final = None
                messages.append({
                    "role": "user",
                    "content": (
                        "validate_code on your final_answer reported issues:\n"
                        + json.dumps(check, ensure_ascii=False)
                        + "\nFix the code and call final_answer again."
                    ),
                })
                continue
            yield {
                "type": "final",
                "code": final.get("code", ""),
                "explanation": final.get("explanation", ""),
            }
            yield {"type": "done"}
            return

    yield {
        "type": "error",
        "message": "Agent did not converge within max_iters.",
    }
    yield {"type": "done"}


def build_llm(args) -> LLM:
    if args.provider == "ollama":
        return get_llm("ollama", model=args.model)
    if args.provider == "gemini":
        model = args.model if args.model and args.model != "gpt-oss:20b" else "gemini-1.5-pro"
        return get_llm("gemini", model=model)
    raise ValueError(args.provider)


def main():
    ap = argparse.ArgumentParser(description="Strudel tool-calling agent.")
    ap.add_argument("prompt", nargs="*", help="One-shot prompt; omit for REPL mode.")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "gemini"])
    ap.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Model id (default gpt-oss:20b for ollama)",
    )
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    llm = build_llm(args)

    def run_once(prompt: str):
        result = run_agent(
            prompt, llm, max_iters=args.max_iters, verbose=args.verbose
        )
        print("\n=== Strudel code ===")
        print(result["code"] or "(no code produced)")
        if result.get("explanation"):
            print("\n=== Notes ===")
            print(result["explanation"])

    if args.prompt:
        run_once(" ".join(args.prompt))
        return

    print(f"Strudel agent ready (provider={args.provider}, model={args.model}).")
    print("Type a prompt, 'exit' to quit.\n")
    while True:
        try:
            user = input("strudel> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            break
        run_once(user)
        print()


if __name__ == "__main__":
    main()
