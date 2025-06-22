import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
chunks = np.load("doc_chunks.npy", allow_pickle=True)
metadata = np.load("doc_metadata.npy", allow_pickle=True)
index = faiss.read_index("faiss_index.bin")

gemini = genai.GenerativeModel("gemini-1.5-pro")

def get_top_chunks(query, top_k=4):
    query_vec = embedding_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

def build_prompt(context_chunks, user_prompt):
    context = "\n---\n".join(context_chunks)
    return f"""You are an expert in Strudel, a live-coding music language.

Below is documentation you can use to answer the user's query:

{context}

---

Now, write Strudel code to fulfill this request:

"{user_prompt}"

Only return Strudel code and a short explanation if necessary.
"""

print("🎼 Strudel Code Generator CLI (powered by Gemini)")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("🧠 Prompt: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break

    top_chunks = get_top_chunks(user_input)
    prompt = build_prompt(top_chunks, user_input)

    print("⚙️ Generating code...\n")
    response = gemini.generate_content(prompt)
    print("🎹 Strudel Code:\n")
    print(response.text)
    print("\n" + "="*60 + "\n")
