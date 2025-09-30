# ===================== resume_processor.py =====================
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize once (LangChain will read OPENAI_API_KEY from env)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ---------- Load a single resume file ----------
def load_resume(file_path: str):
    file_path = str(file_path)
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    return loader.load()  # -> list[Document]

# ---------- Split documents into chunks ----------
def split_docs(docs, chunk_size=1000, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# ---------- Analyze resume text vs job description ----------
def analyze_resume(docs, job_description: str) -> str:
    chunks = split_docs(docs, chunk_size=1000, chunk_overlap=120)

    full_analysis = []
    for chunk in chunks:
        prompt = f"""
Compare this resume with the job description and provide:
1) Suitability score (0-100)
2) Skills matched
3) Experience relevance
4) Education evaluation
5) Strengths
6) Gaps
7) Final recommendation (1-2 sentences)

Job Description:
{job_description}

Resume snippet:
{chunk.page_content}
"""
        resp = llm.invoke(prompt)
        full_analysis.append(resp.content.strip())

    # Simple merge; you could also do a final synthesis pass if desired
    return "\n\n".join(full_analysis)

# ---------- Persist chunks to Chroma ----------
def store_to_vectorstore(docs, persist_directory: str = "chroma_store"):
    persist_dir = Path(persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_docs(docs, chunk_size=1000, chunk_overlap=120)
    texts = [c.page_content for c in chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,  # correct arg name for from_texts
        metadatas=metadatas,
        persist_directory=str(persist_dir),
    )
    vectordb.persist()
    return vectordb

# ---------- Query stored resumes (simple, reliable) ----------
def run_self_query(query: str, persist_directory: str = "chroma_store", k: int = 4):
    # When re-opening a persisted Chroma, pass embedding_function=
    vectordb = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
    # Simple similarity search is stable and effective
    return vectordb.similarity_search(query, k=k)

