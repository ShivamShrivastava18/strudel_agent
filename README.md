### Strudel Agent
# 🥝 Strudel Code Generator CLI 🎼

A command-line AI agent that writes [Strudel](https://strudel.cc/) live-coding music patterns using natural language prompts.

It uses:

* ✅ Web-scraped Strudel documentation
* ✅ Embedding-based search with FAISS
* ✅ Context-aware code generation using **Gemini API**

---

## 📦 Features

* 🔍 Retrieve relevant Strudel documentation via embeddings
* 🌹 Generate accurate Strudel code snippets with natural prompts
* 🧠 Powered by Google Gemini + Sentence Transformers
* 🔤 Fully offline embeddings & fast CPU-friendly inference
* 🪄 Easy-to-use terminal interface

---

## 🚀 Quickstart

### 1. Clone the repo and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your `.env` file

Create a `.env` file and add your Gemini API key:

```
GEMINI_API_KEY=your-api-key-here
```
---

### 3. Scrape the Strudel Docs

Put the links in a `site.txt` file (one URL per line) and run:

```bash
python scrape_docs.py
```

This saves the pages into `Data/`.

---

### 4. Generate Embeddings

```bash
python embed_strudel_docs.py
```

This creates:

* `faiss_index.bin` — vector index for retrieval
* `doc_chunks.npy` — text chunks
* `doc_metadata.npy` — source file tracking

---

### 5. Run the CLI Agent

```bash
python agent.py
```

Now type in natural language prompts, like:

```
🧠 Prompt: play a kick on every beat and hi-hat on 16ths
```

And you’ll get valid Strudel code!

---

## 🧪 Sample Prompts

* Play a kick on every beat and a snare on the off-beats
* Add a hi-hat on 16th notes
* Create a polyrhythm with a triplet snare and 4/4 kick
* Loop a melodic sequence using `n` for notes

---

## 💠 Tech Stack

| Tool                     | Purpose                         |
| ------------------------ | ------------------------------- |
| 🧠 Sentence Transformers | Embedding text chunks           |
| 🔂 FAISS                 | Fast document similarity search |
| 🌐 Gemini API            | Code generation                 |
| 🗞 BeautifulSoup         | HTML scraping                   |
| 🧪 CLI / Python          | Interaction interface           |


---

## ✨ Future Ideas

* 🌍 Streamlit or web UI
* 🎤 Voice-to-music generation
* 🎛️ Real-time live-coding with sound
* 🔄 Add full Strudel workshop docs for broader support

---

## 📝 License

MIT — free to modify and use for learning or art!

---

## ❤️ Made for creative coders & live music lovers
