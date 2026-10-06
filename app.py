"""RepoLens - Streamlit Application Entrypoint.

Provides the web UI for repository indexing, question answering,
retrieval inspection, and parameter configuration.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.loader import load_from_uploaded_files, load_from_directory
from src.rag_pipeline import RepoLensPipeline
from src.llm import is_gemini_configured, get_gemini_api_key

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RepoLens — AI Codebase Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean presentation
st.markdown("""
<style>
    .source-tag {
        display: inline-block;
        background-color: #2e3846;
        color: #e0e6ed;
        padding: 4px 10px;
        margin: 4px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .metric-card {
        padding: 12px;
        background-color: #1e2530;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = RepoLensPipeline()
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = ""


def render_sidebar():
    """Render the sidebar configuration and information panels."""
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("Configure RAG retrieval and chunking parameters.")

    top_k = st.sidebar.slider(
        "Top K Chunks",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of most relevant code chunks to retrieve from FAISS.",
    )
    chunk_size = st.sidebar.slider(
        "Chunk Size (chars)",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
        help="Maximum character length of each chunk before splitting.",
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (chars)",
        min_value=0,
        max_value=400,
        value=100,
        step=20,
        help="Number of shared characters between adjacent chunks to maintain context.",
    )
    show_chunks = st.sidebar.checkbox(
        "Show Retrieved Chunks",
        value=True,
        help="Display the retrieval inspector containing retrieved chunks and similarity scores.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 LLM Model")
    llm_model = st.sidebar.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 API Key Status")
    if is_gemini_configured():
        st.sidebar.success("✅ GEMINI_API_KEY detected in .env")
    else:
        st.sidebar.warning("⚠️ GEMINI_API_KEY is not configured.\nPlease add GEMINI_API_KEY to .env.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏛️ RAG Mental Model")
    st.sidebar.caption(
        "**Retriever:** Finds relevant chunks in FAISS.\\\n"
        "**LLM:** Understands chunks and generates grounded answers.\\\n"
        "**Sources:** Generated from retrieval metadata, not LLM output."
    )

    return {
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "show_chunks": show_chunks,
        "llm_model": llm_model,
    }


def main():
    settings = render_sidebar()
    pipeline: RepoLensPipeline = st.session_state["pipeline"]

    st.title("🔍 RepoLens")
    st.markdown("### AI Codebase & Documentation Assistant")
    st.caption("Upload code & docs, index into local vector database, and ask grounded questions without hallucinations.")

    st.markdown("---")

    # Section 1: Ingest Repository Files
    st.subheader("1. Ingest Repository Files")

    tab_upload, tab_sample = st.tabs(["📁 Upload Local Files", "📦 Load Sample Demo Repo"])

    uploaded_files = []
    load_sample_clicked = False

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Select repository files to index",
            accept_multiple_files=True,
            type=["py", "js", "jsx", "ts", "tsx", "html", "css", "md", "txt", "json", "java", "cpp", "c", "sql"],
            help="Supported: .py, .js, .jsx, .ts, .tsx, .html, .css, .md, .txt, .json, .java, .cpp, .c, .sql. Secrets & ignored folders are automatically filtered.",
        )

    with tab_sample:
        st.markdown("Load pre-configured demo repository (`TaskFlow` sample: `server.js`, `auth.js`, `middleware.js`, `database.js`, `app.js`, `README.md`).")
        load_sample_clicked = st.button("📥 Load Sample Repo Files", key="load_sample_btn")

    col_btn, col_stats = st.columns([1, 2])

    with col_btn:
        index_btn = st.button("🚀 Index Repository", use_container_width=True, type="primary")

    with col_stats:
        status_badge = "🟢 Indexed" if pipeline.is_indexed() else "⚪ Not Indexed"
        st.markdown(
            f"**Repository Status:** {status_badge}  |  "
            f"**Files:** `{pipeline.indexed_files_count}`  |  "
            f"**Chunks:** `{pipeline.indexed_chunks_count}`"
        )

    # Handle Indexing Action
    if index_btn:
        if not uploaded_files:
            st.warning("Please upload repository files first.")
        else:
            with st.spinner("Processing files, generating embeddings, and building FAISS index..."):
                docs = load_from_uploaded_files(uploaded_files)
                if not docs:
                    st.error("No valid supported files found in upload. Supported: .py, .js, .ts, .html, .css, .md, .txt, .json, etc.")
                else:
                    try:
                        stats = pipeline.index_documents(
                            docs,
                            chunk_size=settings["chunk_size"],
                            chunk_overlap=settings["chunk_overlap"],
                        )
                        st.success(
                            f"Successfully indexed {stats['files_indexed']} files into {stats['chunks_created']} chunks!"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to index repository: {e}")

    if load_sample_clicked:
        if not os.path.exists("sample_repo"):
            st.error("Sample repository folder 'sample_repo' not found.")
        else:
            with st.spinner("Indexing sample repository files into FAISS..."):
                try:
                    stats = pipeline.index_from_directory(
                        "sample_repo",
                        chunk_size=settings["chunk_size"],
                        chunk_overlap=settings["chunk_overlap"],
                    )
                    st.success(
                        f"Sample repository indexed: {stats['files_indexed']} files, {stats['chunks_created']} chunks!"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to index sample repo: {e}")

    st.markdown("---")

    # Section 2: Q&A Interface
    st.subheader("2. Ask RepoLens")

    # Interview Demo Prompts Quick-Select
    st.markdown("**Quick Demo Questions:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    if q_col1.button("📋 What does project do?", use_container_width=True):
        st.session_state["selected_query"] = "What does this project do?"
    if q_col2.button("🔐 Authentication flow?", use_container_width=True):
        st.session_state["selected_query"] = "How is authentication implemented?"
    if q_col3.button("🗄️ Database connection?", use_container_width=True):
        st.session_state["selected_query"] = "Which file handles database connection?"
    if q_col4.button("💳 Payment gateway test", use_container_width=True):
        st.session_state["selected_query"] = "How does this project process credit card payments?"

    current_query_value = st.session_state["selected_query"]
    user_query = st.text_input(
        "Enter your question about the repository:",
        value=current_query_value,
        placeholder="e.g., How is authentication implemented? Which file handles database connection?",
    )

    ask_btn = st.button("🤖 Ask AI", use_container_width=False)

    if ask_btn:
        if not user_query.strip():
            st.warning("Please enter a question.")
        elif not pipeline.is_indexed():
            st.error("Please index the repository before asking questions.")
        elif not is_gemini_configured():
            st.error("Gemini API key is not configured. Please add GEMINI_API_KEY to .env.")
        else:
            with st.spinner("Retrieving relevant code chunks and generating answer via Gemini..."):
                result = pipeline.query(
                    question=user_query,
                    top_k=settings["top_k"],
                    llm_model=settings["llm_model"],
                )
                st.session_state["last_result"] = result

    # Display Query Results
    if st.session_state["last_result"]:
        result = st.session_state["last_result"]

        st.markdown("---")
        st.subheader("3. Answer")
        st.markdown(result["answer"])

        st.markdown("---")
        st.subheader("4. Sources")
        sources = result.get("sources", [])
        if sources:
            st.caption("Sources retrieved strictly from vector store chunk metadata (not hallucinated):")
            for idx, src in enumerate(sources, start=1):
                st.markdown(f"**{idx}.** `{src}`")
        else:
            st.write("No matching source files retrieved.")

        # Section 5: Retrieval Inspector (Interview Feature)
        if settings["show_chunks"]:
            st.markdown("---")
            with st.expander("🔍 Retrieved Context (RAG Inspection)", expanded=True):
                st.markdown(
                    "Demonstrates the actual chunks retrieved from FAISS before being sent to Gemini:"
                )
                retrieved_chunks = result.get("retrieved_chunks", [])
                if not retrieved_chunks:
                    st.write("No chunks retrieved.")
                else:
                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(
                            f"**Chunk {i} — `{chunk['source']}`** &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"*L2 Distance Score:* `{chunk['score']}` &nbsp;&nbsp;|&nbsp;&nbsp; "
                            f"*Type:* `{chunk['file_type']}`"
                        )
                        st.code(chunk["content"], language=chunk.get("file_type", "text"))
                        st.markdown("---")


if __name__ == "__main__":
    main()
