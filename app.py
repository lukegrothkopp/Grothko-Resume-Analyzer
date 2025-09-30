import os
import tempfile
from pathlib import Path
import streamlit as st

from resume_processor import (
    load_resume,
    analyze_resume,
    store_to_vectorstore,
    run_self_query,
)

st.set_page_config(page_title="AI Resume Screener", page_icon="🧠", layout="wide")
st.title("AI Resume Screener")
st.caption("Upload a resume, analyze it against a job description, and search across stored resumes.")

# --- Key check (works with Streamlit Secrets or local .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add it in Streamlit Secrets or .env for full functionality.")

# --- Inputs ---
job_desc = st.text_area("📋 Paste Job Description", height=180, placeholder="Paste the job description here...")
uploaded_file = st.file_uploader("📎 Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

colA, colB = st.columns([1, 1])

with colA:
    if st.button("Analyze & Store", use_container_width=True) and uploaded_file and job_desc:
        # Save uploaded file to a temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        with st.spinner("Analyzing & storing resume..."):
            docs = load_resume(tmp_path)
            report = analyze_resume(docs, job_desc)
            store_to_vectorstore(docs, persist_directory="chroma_store")

        st.success("✅ Analysis complete and stored.")
        st.subheader("📄 AI Resume Summary")
        st.write(report)
        st.download_button("📥 Download Report", report, file_name="resume_analysis.txt")

with colB:
    st.subheader("🔎 Ask Anything About Stored Resumes")
    query = st.text_input("Enter a smart query (e.g., 'Python developer with AWS and Kubernetes')")
    if st.button("Search Resumes", use_container_width=True) and query:
        with st.spinner("Searching..."):
            results = run_self_query(query, persist_directory="chroma_store", k=5)
        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Result {i}**")
                st.write(res.page_content.strip())
                meta = res.metadata or {}
                if meta:
                    st.caption(f"Source: {meta.get('source','unknown')}")
                st.divider()
        else:
            st.warning("No matches found. Try a broader query or upload more resumes.")
