# graph_rag.py
import os
from typing import List, Dict, Set
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- CONFIG ----------
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast embedder
GEN_MODEL = "google/flan-t5-small" # small free generator (CPU friendly)
DOCS_PATH = "data/docs.txt"
CHUNK_SIZE = 300      # characters per chunk (tune)
TOP_K = 5             # initial vector retrieval
GRAPH_K = 3           # connect each node to k nearest neighbors
EXPAND_HOPS = 1       # graph expansion hops
MAX_CONTEXT_CHUNKS = 6  # number of chunks to pass to generator
# ----------------------------

def load_docs(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

def chunk_doc(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Simple fixed-length chunking by characters (keeps it deterministic)."""
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        chunk = text[start:start+chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size
    return chunks

def build_corpus(docs: List[str]) -> List[Dict]:
    corpus = []
    for i, d in enumerate(docs):
        chunks = chunk_doc(d)
        for j, c in enumerate(chunks):
            corpus.append({"id": f"doc{i}_chunk{j}", "text": c, "source_doc": i})
    return corpus

def embed_texts(model, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    # normalize vectors for cosine similarity via inner product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    embs = embs / norms
    return embs.astype("float32")

def build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine sim
    index.add(embs)
    return index

def build_semantic_graph(corpus: List[Dict], embeddings: np.ndarray, k: int = GRAPH_K):
    G = nx.Graph()
    n = len(corpus)
    sims = cosine_similarity(embeddings)  # n x n
    for i, node in enumerate(corpus):
        G.add_node(i, **node)
    # connect top-k neighbors (skip self)
    for i in range(n):
        row = sims[i]
        idxs = np.argsort(-row)
        added = 0
        for idx in idxs:
            if idx == i: continue
            G.add_edge(i, idx, weight=float(row[idx]))
            added += 1
            if added >= k:
                break
    return G

def retrieve_then_expand(query: str, embed_model, index, corpus, embeddings, G, top_k=TOP_K, hops=EXPAND_HOPS):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    q_emb = q_emb.astype("float32")
    D, I = index.search(q_emb, top_k)
    seed_idxs = [int(x) for x in I[0] if x != -1]
    # expand
    result_set: Set[int] = set(seed_idxs)
    frontier = set(seed_idxs)
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in result_set:
                    new_frontier.add(nb)
        result_set.update(new_frontier)
        frontier = new_frontier
    # collect and rerank by cosine to query
    candidates = []
    for idx in result_set:
        cand = corpus[idx]
        emb = embeddings[idx:idx+1]
        sim = float(np.dot(q_emb, emb.T))
        candidates.append((idx, sim, cand["text"], cand["id"]))
    candidates.sort(key=lambda x: -x[1])
    return candidates

def format_context_chunks(chunks: List[str]) -> str:
    # Simple join with separators
    return "\n\n---\n\n".join(chunks)

def generate_answer_local(generator, tokenizer, query: str, contexts: List[str]):
    # Build prompt for Flan-T5 style: give context then question and instruction
    context_text = format_context_chunks(contexts)
    prompt = (
        "Use the following passages to answer the question concisely. "
        "If the answer is not present, say 'I don't know from the provided context.'\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    )
    # use generator pipeline (text2text) — limit max_length to something modest
    out = generator(prompt, max_length=256, do_sample=False)
    return out[0]['generated_text']

def main():
    print("Loading docs...")
    docs = load_docs(DOCS_PATH)
    corpus = build_corpus(docs)
    texts = [c["text"] for c in corpus]
    print(f"Built {len(texts)} chunks from {len(docs)} docs.")
    print("Loading embedder...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_texts(embed_model, texts)
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    print("Building semantic graph...")
    G = build_semantic_graph(corpus, embeddings, k=GRAPH_K)

    print("Loading generator model (this may take a moment)...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

    print("\nReady. Type queries (type 'exit' to quit).")
    while True:
        q = input("\nQuery> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        candidates = retrieve_then_expand(q, embed_model, index, corpus, embeddings, G, top_k=TOP_K, hops=EXPAND_HOPS)
        if not candidates:
            print("No candidates found in corpus.")
            continue
        # take top-N contexts
        top = candidates[:MAX_CONTEXT_CHUNKS]
        context_texts = [c[2] for c in top]
        answer = generate_answer_local(generator, tokenizer, q, context_texts)
        print("\n=== Answer ===\n")
        print(answer.strip())
        print("\n=== Sources (top candidates) ===")
        for idx, score, text, cid in top:
            print(f"[{cid}] score={score:.3f} -> {text[:200].replace('\\n',' ')}")
    print("Goodbye.")

if __name__ == "__main__":
    main()

# To Run:
# .venv\Scripts\activate
# python graph_rag.py
