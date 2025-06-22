import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === CONFIG ===
DOC_DIR = "Data"
CHUNK_SIZE = 500
OVERLAP = 100
BATCH_SIZE = 16
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"

# === Load model ===
print(f"📦 Loading model '{MODEL_NAME}' on {DEVICE}...")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# === Chunking helper ===
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === Read and chunk all docs ===
print(f"📄 Reading and chunking docs in '{DOC_DIR}'...")
all_chunks = []
metadata = []

for filename in sorted(os.listdir(DOC_DIR)):
    if filename.endswith(".txt"):
        path = os.path.join(DOC_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = chunk_text(content, CHUNK_SIZE, OVERLAP)
            all_chunks.extend(chunks)
            metadata.extend([filename] * len(chunks))

print(f"🧠 Total chunks: {len(all_chunks)}")

# === Embedding in batches ===
print("🔄 Generating embeddings (CPU)...")
embeddings = []

for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
    batch = all_chunks[i:i + BATCH_SIZE]
    embs = model.encode(batch)
    embeddings.extend(embs)

embeddings = np.array(embeddings)
print(f"✅ Embeddings shape: {embeddings.shape}")

# === Build FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Save to disk ===
faiss.write_index(index, "faiss_index.bin")
np.save("doc_chunks.npy", all_chunks)
np.save("doc_metadata.npy", metadata)

print("🎉 All files saved:")
print(" - FAISS index: faiss_index.bin")
print(" - Chunks: doc_chunks.npy")
print(" - Metadata: doc_metadata.npy")
