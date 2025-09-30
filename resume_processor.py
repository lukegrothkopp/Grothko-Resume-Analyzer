# ===================== resume_processor.py =====================
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Load resumes in different formats
def load_resume(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        import docx2txt
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# Analyze the resume using LLM
def analyze_resume(docs, job_description):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    full_analysis = ""
    for chunk in chunks:
        prompt = f"""
Compare this resume with the job description. Give:
1. Suitability Score (out of 100)
2. Skills Matched
3. Experience Relevance
4. Education Evaluation
5. Strengths
6. Weaknesses
7. Final Recommendation

Job Description:
{job_description}

Resume:
{chunk.page_content}
"""
        result = llm.invoke(prompt)
        full_analysis += result.content + "\n\n"
    return full_analysis

# Store text chunks into ChromaDB
def store_to_vectorstore(text_chunks, persist_directory="chroma_store"):
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# Use SelfQueryRetriever to interpret and fetch relevant chunks
def run_self_query(query, persist_directory="chroma_store"):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Where the chunk is from",
            type="string"
        )
    ]

    document_content_description = "This represents a chunk of a resume."

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        search_type="mmr"
    )

    return retriever.get_relevant_documents(query)
