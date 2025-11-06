import os
import tempfile
from pathlib import Path
import streamlit as st

from resume_processor import (
    load_resume,
    analyze_resume,
    store_to_vectorstore,
    run_self_query,
    compute_fit_scores,
    store_job_description,
)

# Resolve repo-relative path to the logo
REPO_DIR = Path(__file__).parent
LOGO_PATH = REPO_DIR / "assets" / "header_logo.png"   # <-- your file in the repo

# Helper: return a usable icon value for Streamlit (path if exists, else emoji)
def page_icon_value():
    return str(LOGO_PATH) if LOGO_PATH.exists() else "🧠"

st.set_page_config(page_title="Grothko AI Resume Screener", page_icon=page_icon_value(), layout="wide")
st.markdown("""
<style>
/* Give the main container more breathing room at the top */
.block-container { padding-top: 4.5rem !important; }

/* (optional) nudge the header down a touch more */
.app-header { margin-top: 6px; }

/* Mobile: slightly less padding so you don't waste space */
@media (max-width: 680px) {
  .block-container { padding-top: 3.2rem !important; }
}
</style>
""", unsafe_allow_html=True)

logo_src = str(LOGO_PATH) if LOGO_PATH.exists() else None

# Build header row
c1, c2 = st.columns([0.08, 0.92])
with c1:
    if logo_src:
        st.image(logo_src, width=42)
    else:
        st.markdown("🧠", unsafe_allow_html=True)
with c2:
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="app-title">Grothko AI Resume Screener</div>
                <div class="app-sub">Upload a resume, analyze it against a job description, filter relevant chunks, and search across stored resumes.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Key check (works with Streamlit Secrets or local .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Add it in Streamlit Secrets or .env for full functionality.")

# --- Inputs ---
job_title = st.text_input("📝 Job Title / Label", value="Software Engineer")
job_desc = st.text_area("📋 Paste Job Description", height=200, placeholder="Paste the job description here...")

uploaded_file = st.file_uploader("📎 Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

st.markdown("### Storage Options")
col_filter1, col_filter2, col_filter3 = st.columns([1,1,1], vertical_alignment="center")

with col_filter1:
    only_best = st.checkbox("Store only best chunks (by similarity to JD)", value=True)
with col_filter2:
    top_k = st.slider("Top-K chunks to keep", min_value=1, max_value=40, value=8, step=1, disabled=not only_best)
with col_filter3:
    min_sim = st.slider("Min similarity (0–1)", min_value=0.00, max_value=1.00, value=0.00, step=0.01, disabled=not only_best)

if not only_best:
    top_k = 0
    min_sim = None

colA, colB = st.columns([1, 1])

with colA:
    analyze_clicked = st.button("Analyze & Store", use_container_width=True)

if analyze_clicked and uploaded_file and job_desc:
    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Analyzing resume..."):
        docs = load_resume(tmp_path)
        chunk_analyses, final_report = analyze_resume(docs, job_desc)

    # Optional: store job description (for organization/traceability)
    store_job_description(job_desc, label=job_title, persist_directory="chroma_jd")

    # Compute vector-based fit score vs JD
    fit = compute_fit_scores(job_desc, docs, top_k=5)

    # Store resume chunks with optional filtering
    with st.spinner("Storing resume chunks..."):
        stats = store_to_vectorstore(
            docs,
            persist_directory="chroma_store",
            filter_job_description=job_desc if only_best else None,
            top_k=top_k if only_best else 0,
            min_sim=min_sim if only_best else None,
        )

    st.success("✅ Analysis complete and stored.")
    st.subheader("📊 Vector Fit Score")
    col1, col2, col3 = st.columns(3)
    col1.metric("Suitability (vector, 0–100)", fit["vector_suitability_0_100"])
    col2.metric("Max similarity", f"{fit['max_sim']:.3f}")
    col3.metric("Avg top-K similarity", f"{fit['avg_top_k_sim']:.3f}")

    with st.expander("Top matches (vector search)", expanded=False):
        for i, (snippet, sim) in enumerate(fit["top_matches"], 1):
            st.markdown(f"**Match {i} — sim {sim:.3f}**")
            st.write(snippet.strip())
            st.divider()

    st.subheader("📄 Final Consolidated Report")
    st.write(final_report)
    st.download_button("📥 Download Report", final_report, file_name=f"{job_title.replace(' ','_')}_resume_analysis.txt")

    with st.expander("Per-chunk analyses (raw)", expanded=False):
        for i, txt in enumerate(chunk_analyses, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(txt)
            st.divider()

st.divider()
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
                sim_txt = f" | sim≈{meta.get('jd_similarity'):.3f}" if "jd_similarity" in meta else ""
                st.caption(f"Source: {meta.get('source','unknown')}{sim_txt}")
            st.divider()
    else:
        st.warning("No matches found. Try a broader query or upload more resumes.")
