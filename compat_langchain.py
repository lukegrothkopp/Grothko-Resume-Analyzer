# compat_langchain.py
# Centralizes imports across LangChain 0.0/0.1.x vs 0.2.x+

# Text splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # >=0.2
except Exception:  # old path
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # <=0.1

# Vectorstores (FAISS)
try:
    from langchain_community.vectorstores import FAISS  # >=0.2
except Exception:
    from langchain.vectorstores import FAISS  # <=0.1

# Loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
    )
except Exception:
    # Older locations
    try:
        from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    except Exception:
        # Some older builds call it Unstructured* or use PyPDF2; add more fallbacks if needed
        PyPDFLoader = None
        Docx2txtLoader = None
        TextLoader = None

# Embeddings / LLMs
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # modern
except Exception:
    # Very old style – if you were using OpenAI directly; keep modern preferred
    OpenAIEmbeddings = None
    ChatOpenAI = None
