# Strudel Agent

A tool-calling LLM agent that generates **validated** [Strudel](https://strudel.cc/)
live-coding music patterns from natural-language prompts.

The agent grounds every generation in a curated Strudel registry, lets the
model search the docs and look up function signatures via tools, and runs a
syntax/semantics validator on every candidate snippet before returning it —
so the output uses **only real Strudel functions and samples**.

```
strudel> kick on every beat with a clap on 2 and 4, hi-hat on 8ths

=== Strudel code ===
stack(
  s("bd*4"),
  s("~ cp ~ cp"),
  s("hh*8")
)
```

---

## Why a tool-calling agent?

Naive RAG generators frequently hallucinate Strudel APIs (`reverb()`,
`filter()`, made-up samples). This agent fixes that with four layers of
defence:

1. **Curated registry** (`strudel_registry.json`) — ground truth list of
   ~120 Strudel functions, drum-machine banks, sample names, signals, and
   common pitfalls.
2. **Vibe catalog** (`styles.json` + `vibe.py`) — turns vague prompts
   ("make it sound like Daft Punk", "rainy night") into concrete musical
   specs (tempo BPM, key, drums, bass, melody, palette samples, drum
   machine) so the model never has to guess what an artist sounds like.
3. **Tools the model must use** — `interpret_vibe`, `web_search_vibe`,
   `lookup_function`, `list_samples`, `get_examples`, `search_docs`,
   `validate_code`, `final_answer`.
4. **Validator-in-the-loop** — every candidate snippet is checked for
   unknown methods, unknown samples, raw strings passed to `stack()`/`cat()`,
   note names mistakenly given to `s()`, etc. Issues are fed back to the
   model until validation passes.

---

## Architecture

```
                 user prompt
                      |
                      v
          ┌───────────────────────┐
          │   agent.py loop       │
          │  (system prompt +     │
          │  tool-calling)        │
          └─────────┬─────────────┘
                    │
   ┌──────────┬─────┴─────┬─────────────┬───────────┐
   v          v           v             v           v
 search     lookup     get          validate     final
 _docs    _function    _examples    _code        _answer
   │          │           │             │
   │          │           │             │
   v          v           v             v
 FAISS    registry.py  examples.    registry.py
 index    + json       jsonl        validator
```

Files:
| File | Purpose |
| ---- | ------- |
| `agent.py` | Tool-calling REPL/CLI |
| `llm.py` | Provider abstraction (Ollama / Gemini) with tool calling |
| `tools.py` | Tool implementations + JSON schemas exposed to the model |
| `registry.py` | Validator + lookups against `strudel_registry.json` |
| `strudel_registry.json` | Curated catalog of valid Strudel identifiers |
| `examples.jsonl` | Hand-verified working snippets (retrieved by `get_examples`) |
| `vibe.py` | Vague-prompt detector + style spec interpreter (Tavily fallback) |
| `styles.json` | Curated catalog of artists / genres / moods / scenes |
| `server.py` | FastAPI playback server (SSE agent stream + strudel.cc iframe) |
| `web/` | Frontend (`index.html`, `style.css`, `app.js`) for the playback app |
| `embed_strudel_docs.py` | Semantic-aware chunker + FAISS index builder |
| `scraping.py` | Doc scraper |

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull a local model with tool support (default)

The agent defaults to **gpt-oss:20b** via [Ollama](https://ollama.com/) —
~13 GB, fits comfortably in 16 GB+ RAM, supports tool calling.

```bash
ollama pull gpt-oss:20b
ollama serve   # if not already running
```

### 3. (Optional) (Re-)build the doc index

If you want fresh docs:

```bash
python scraping.py            # writes Data/page_XX.txt
python embed_strudel_docs.py  # writes faiss_index.bin, doc_chunks.npy, doc_metadata.npy
```

The repo already ships with a pre-built FAISS index, so this step is
optional.

### 4. Run the agent

```bash
# Interactive REPL
python agent.py

# One-shot
python agent.py "minimal techno loop with sub bass and offbeat hats"

# Verbose (see thoughts + tool calls)
python agent.py -v "amen-style breakbeat with a filter sweep"

# Use Gemini instead (needs GEMINI_API_KEY in .env — note: tool-calling
# behaviour is best with Ollama)
python agent.py --provider gemini
```

CLI flags:
- `--provider {ollama,gemini}` — default `ollama`
- `--model NAME` — default `gpt-oss:20b`
- `--max-iters N` — tool-calling loop budget (default 8)
- `-v / --verbose` — print thinking and tool traces

### 5. Run the web playback app

```bash
python server.py            # serves http://127.0.0.1:8000
# or with auto-reload during development
python server.py --reload --port 8000
```

Open `http://127.0.0.1:8000` in a browser. The UI has three panes:

- **Prompt box** (top-left): type a request, hit *Generate*.
- **Agent thinking stream** (left): tool calls, validator results, vibe
  matches, and the model's reasoning are streamed live via SSE.
- **Code editor + strudel REPL** (right): the final code lands in an
  editable textarea. Click **▶ Play / reload** to load it into an
  embedded `strudel.cc` iframe (which has its own play/stop controls).
  *↗ Open in strudel.cc* opens the same code in a new tab.

Live-edit validation runs as you type — green check = registry-valid,
red cross = unknown function/sample (with suggestions). The page hits
`POST /api/generate` (SSE) and `POST /api/validate` against the same
agent backend.

Server env overrides (optional):
```bash
export STRUDEL_AGENT_PROVIDER=ollama       # or gemini
export STRUDEL_AGENT_MODEL=gpt-oss:20b
export STRUDEL_AGENT_MAX_ITERS=10
```

---

## Sample prompts

Specific (music-theory) prompts:

```
kick on every beat with a clap on 2 and 4, hi-hat on 8ths
acid bassline in C minor with a resonant filter sweep
arpeggiated I-vi-IV-V chord progression with reverb
amen breakbeat with degraded hats
polyrhythmic 3-against-4 percussion
FM bell-like melody on a pentatonic scale
```

Vague / vibe prompts (auto-translated via `styles.json`):

```
make it sound like daft punk
joji style beat
rainy night studying
i'm sad and want something slow
gym pump-up energy
boards of canada vibe
```

The vibe layer covers ~50 entries spanning artists (Daft Punk, Joji,
Aphex Twin, BoC, Burial, J Dilla, Tycho, Four Tet, Radiohead, …),
genres (lo-fi, trap, house, techno, dnb, ambient, vaporwave, synthwave,
jazz, drill, …), moods (sad, happy, energetic, dreamy, dark, chill,
…), and scenes (study beat, rainy night, driving at night, party,
horror movie, …). Each entry expands into a concrete spec — tempo BPM
range, key hint, drum-machine bank, palette samples/synths, and feel —
which is auto-injected ahead of the user's prompt so the model never
has to guess.

---

## Vibe catalog

`styles.json` is structured as four top-level dicts: `artists`,
`genres`, `moods`, `scenes`. Each entry looks like:

```json
{
  "joji": {
    "aliases": ["joji george"],
    "genres": ["lo-fi", "alt r&b", "sad pop"],
    "tempo_bpm": [70, 95],
    "key_hint": "minor (often C# minor, F minor)",
    "drums": "lazy trap-leaning kick, snappy clap/snare on 3, occasional rolled hats",
    "bass": "deep sub sine bass on root; sparse",
    "melody": "soft sine/triangle leads, melancholic lines with bends and small intervals",
    "fx": "heavy reverb, lo-pass filtering, vinyl crackle, slow chorus",
    "palette_samples": ["bd", "cp", "hh", "rim"],
    "palette_synths": ["sine", "triangle"]
  }
}
```

Add entries by appending to `styles.json` — no code changes needed.
Scenes can `"based_on": "lo-fi"` to inherit a genre and override a few
fields.

When the user asks for an artist/style not in the catalog, the agent
calls `web_search_vibe` (Tavily) for a one-shot description. Set your
Tavily key in `.env`:

```bash
cp .env.example .env
# then edit .env and add TAVILY_API_KEY=tvly-...
```

If `TAVILY_API_KEY` is unset, the web fallback is silently skipped and
the agent leans on its own knowledge.

---

## Tech stack

| Component | Purpose |
| --------- | ------- |
| Ollama (gpt-oss:20b) | Local tool-calling LLM |
| sentence-transformers | Embeddings (`all-MiniLM-L6-v2`) |
| FAISS | Doc retrieval |
| rank-bm25 | Optional hybrid search |
| BeautifulSoup | Doc scraping |
| Gemini (optional) | Cloud fallback provider |
| FastAPI + sse-starlette | Playback server with streaming agent events |
| strudel.cc iframe | Browser playback (no local audio engine needed) |

---

## Roadmap

- [x] Local web playback app (FastAPI + strudel.cc iframe) so the agent
      can actually play what it writes
- [ ] Self-hosted playback via `@strudel/web` (drop the iframe dependency)
- [ ] Hybrid retrieval (BM25 + dense, RRF fusion) in `tools.search_docs`
- [ ] Auto-extract registry deltas from new doc pulls
- [ ] Multi-turn refinement: "make it darker", "add a snare"
- [ ] Expand `styles.json` (more artists, region-specific scenes)
- [ ] Voice-to-music

---

## License

MIT.
