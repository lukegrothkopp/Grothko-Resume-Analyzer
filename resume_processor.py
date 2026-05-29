# ===================== resume_processor.py =====================
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from dotenv import load_dotenv

# Use our compatibility shim so the code works across LangChain versions
from compat_langchain import (
    RecursiveCharacterTextSplitter,
    FAISS,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    OpenAIEmbeddings,
    ChatOpenAI,
)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional: fail fast if missing
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# ---- Global LLM + Embeddings ----
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ----------------- utilities -----------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def split_docs(docs, chunk_size=1000, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# ----------------- loaders -----------------
def load_resume(file_path: str):
    path = str(file_path).lower()
    if path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    return loader.load()  # -> list[Document]

# ----------------- analysis -----------------
def analyze_resume(docs, job_description: str) -> Tuple[List[str], str]:
    """
    Returns:
      chunk_analyses: list of per-chunk analysis strings
      final_synthesis: one consolidated, deduplicated report
    """
    chunks = split_docs(docs, chunk_size=1000, chunk_overlap=120)

    chunk_analyses: List[str] = []
    for chunk in chunks:
        prompt = f"""
Compare this resume snippet with the job description and provide:
1) Suitability score (0-100, integer)
2) Skills matched (bullet list)
3) Experience relevance (2-4 bullets)
4) Education evaluation (1-2 bullets)
5) Strengths (2-4 bullets)
6) Gaps (2-4 bullets)
7) Final recommendation (1-2 sentences)

Job Description:
{job_description}

Resume snippet:
{chunk.page_content}
"""
        resp = llm.invoke(prompt)
        chunk_analyses.append(resp.content.strip())

    # ---- final synthesis step ----
    synthesis_prompt = f"""
You are a hiring assistant. You are given:
- A job description
- Multiple chunk-by-chunk analyses of a candidate's resume

Produce ONE consolidated, deduplicated report with these sections:
1) Overall suitability score (0-100)
2) Key matched skills
3) Notable experience aligned to the role
4) Education evaluation
5) Strengths
6) Gaps or risks
7) Actionable next steps (bullets)
8) Final recommendation (2-3 sentences)

Be concise but specific. Remove duplicates and contradictions. Where relevant, quantify.

Job Description:
{job_description}

Chunk Analyses (raw):
{chr(10).join(f"- {a}" for a in chunk_analyses)}
"""
    final = llm.invoke(synthesis_prompt).content.strip()
    return chunk_analyses, final

# ----------------- persistence helpers (FAISS local index) -----------------
def _faiss_index_exists(persist_dir: Path) -> bool:
    """FAISS save_local creates index.faiss and index.pkl."""
    return (persist_dir / "index.faiss").exists() and (persist_dir / "index.pkl").exists()


def _load_or_create_faiss(texts: List[str], metadatas: List[Dict[str, Any]], persist_dir: Path):
    """
    Load an existing FAISS index and append texts, or create a new one.
    allow_dangerous_deserialization=True is required by LangChain because FAISS
    stores the local docstore in a pickle file. Only load indexes created by this app.
    """
    if _faiss_index_exists(persist_dir):
        vectordb = FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectordb.add_texts(texts=texts, metadatas=metadatas)
    else:
        vectordb = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )

    vectordb.save_local(str(persist_dir))
    return vectordb


# ----------------- persistence (resume chunks) -----------------
def store_to_vectorstore(
    docs,
    persist_directory: str = "faiss_store",
    filter_job_description: str = None,
    top_k: int = 0,
    min_sim: float = None
) -> Dict[str, Any]:
    """
    Stores resume chunks in a local FAISS index with optional filtering based on
    similarity to the job description using OpenAI embeddings.

    filter_job_description: if provided, chunks are ranked by cosine similarity to JD
    top_k: keep only the top_k most similar chunks (if > 0)
    min_sim: keep only chunks with cosine similarity >= min_sim (0-1)

    Returns stats: {"stored":N, "total":M, "filtered":M-N, "avg_sim":..., "min_sim_kept":...}
    """
    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    original_chunks = split_docs(docs, chunk_size=1000, chunk_overlap=120)
    chunks = original_chunks
    texts = [c.page_content for c in chunks]

    if not texts:
        return {"stored": 0, "total": 0, "filtered": 0}

    sims = None
    kept_idx = list(range(len(texts)))

    if filter_job_description:
        # Embed JD and chunks
        q_vec = np.array(embeddings.embed_query(filter_job_description), dtype=float)
        d_vecs = np.array(embeddings.embed_documents(texts), dtype=float)

        # Similarity per chunk
        sims = np.array([cosine_sim(q_vec, d) for d in d_vecs], dtype=float)

        # Apply filters (min_sim first, then top_k)
        idxs = np.arange(len(texts))
        if min_sim is not None:
            idxs = idxs[sims >= float(min_sim)]

        if top_k and top_k > 0 and len(idxs) > top_k:
            # Sort remaining by similarity desc and take top_k
            idxs = idxs[np.argsort(-sims[idxs])[:top_k]]

        kept_idx = list(map(int, idxs))

        # Reduce texts/chunks to kept
        texts = [texts[i] for i in kept_idx]
        chunks = [chunks[i] for i in kept_idx]

    if not texts:
        return {
            "stored": 0,
            "total": len(original_chunks),
            "filtered": len(original_chunks),
        }

    metadatas = []
    for i, c in enumerate(chunks):
        base = {"source": f"resume_chunk_{i}"}
        if sims is not None:
            base["jd_similarity"] = float(sims[kept_idx[i]])
        metadatas.append(base)

    _load_or_create_faiss(texts, metadatas, persist_dir)

    stats = {
        "stored": len(texts),
        "total": len(original_chunks),
        "filtered": len(original_chunks) - len(texts),
    }
    if sims is not None and len(kept_idx) > 0:
        kept_sims = sims[kept_idx]
        stats["avg_sim"] = float(np.mean(kept_sims))
        stats["min_sim_kept"] = float(np.min(kept_sims))
        stats["max_sim_kept"] = float(np.max(kept_sims))
    return stats


# ----------------- job description store -----------------
def store_job_description(
    job_description: str,
    label: str = "Job",
    persist_directory: str = "faiss_jd"
) -> Dict[str, Any]:
    """
    Stores the full job description as a single doc in its own local FAISS index.
    """
    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    texts = [job_description]
    metadatas = [{"label": label, "source": "job_description"}]

    _load_or_create_faiss(texts, metadatas, persist_dir)

    return {"stored": 1, "label": label}

# ----------------- cross-search fit score -----------------
def compute_fit_scores(
    job_description: str,
    docs,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Computes a simple vector-based fit score between a job description and
    resume chunks. Returns:
      - max_sim
      - avg_top_k_sim
      - top_matches: [(snippet, sim), ...]
      - vector_suitability_0_100: int scaled score
    """
    chunks = split_docs(docs, chunk_size=1000, chunk_overlap=120)
    texts = [c.page_content for c in chunks]

    q_vec = np.array(embeddings.embed_query(job_description), dtype=float)
    d_vecs = np.array(embeddings.embed_documents(texts), dtype=float)

    sims = np.array([cosine_sim(q_vec, d) for d in d_vecs], dtype=float)
    order = np.argsort(-sims)
    k = min(top_k, len(texts))
    topk_idx = order[:k]
    top_matches = [(texts[i], float(sims[i])) for i in topk_idx]

    max_sim = float(np.max(sims)) if len(sims) else 0.0
    avg_top_k = float(np.mean(sims[topk_idx])) if k > 0 else 0.0

    # Scale similarity to 0–100 (clip to [0,1] first)
    vector_score = int(round(100 * float(np.clip(avg_top_k, 0.0, 1.0))))

    return {
        "max_sim": max_sim,
        "avg_top_k_sim": avg_top_k,
        "top_matches": top_matches,
        "vector_suitability_0_100": vector_score,
    }

# ----------------- simple search on stored resumes -----------------
from langchain_core.documents import Document


def run_self_query(query: str, persist_directory: str = "faiss_store", k: int = 4) -> List["Document"]:
    persist_dir = Path(persist_directory)

    # If no index dir yet, just say "no results"
    if not _faiss_index_exists(persist_dir):
        return []

    vectordb = FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectordb.similarity_search(query, k=k)
