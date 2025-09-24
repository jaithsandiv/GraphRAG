# Repository Description

**GraphRAG: Lightweight Graph-Enhanced Retrieval-Augmented Generation**

A CPU-friendly implementation of GraphRAG that improves document retrieval by combining vector similarity search with semantic graph expansion. Uses lightweight models (SentenceTransformers + Flan-T5) for local execution without GPU requirements.

## Key Features
🔍 **Hybrid Retrieval** - Vector search + graph expansion for better context  
🧠 **Semantic Graph** - Automatically connects related document chunks  
💻 **CPU-Optimized** - Runs locally with lightweight models  
⚡ **Fast Setup** - Simple pip install and ready to use  
📚 **Educational** - Clean, readable implementation for learning GraphRAG concepts  

## Quick Start
```bash
pip install sentence-transformers networkx scikit-learn faiss-cpu transformers
python graph_rag.py
```

Perfect for researchers, students, and developers wanting to understand and experiment with graph-enhanced RAG systems without heavy computational requirements.
