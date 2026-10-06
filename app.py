"""RepoLens - Streamlit Application Entrypoint.

Provides the web UI for repository indexing, question answering,
retrieval inspection, and parameter configuration.
Supports indexing whole repository folders, uploaded ZIP archives, or individual files.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.loader import (
    load_from_uploaded_files,
    load_from_directory,
    load_from_zip,
    scan_directory_summary,
)
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
    .file-badge {
        font-family: monospace;
        background-color: #1e2633;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
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
if "folder_path_input" not in st.session_state:
    st.session_state["folder_path_input"] = "sample_repo"


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
    st.caption("Index entire repository folders, local projects, or ZIP archives into local FAISS and ask grounded code questions.")

    st.markdown("---")

    # Section 1: Ingest Repository
    st.subheader("1. Ingest Repository")

    # Top Status Bar
    status_badge = "🟢 Indexed" if pipeline.is_indexed() else "⚪ Not Indexed"
    st.info(
        f"**Repository Status:** {status_badge}  |  "
        f"**Indexed Files:** `{pipeline.indexed_files_count}`  |  "
        f"**Chunks in Vector DB:** `{pipeline.indexed_chunks_count}`"
    )

    tab_folder, tab_zip, tab_files = st.tabs([
        "📂 Local Repository Folder",
        "🗜️ Upload Repository ZIP (.zip)",
        "📄 Upload Loose Files",
    ])

    # --- TAB 1: LOCAL REPOSITORY FOLDER ---
    with tab_folder:
        st.markdown("Provide the local folder path to any repository on your machine. All subfolders and supported code files will be indexed automatically.")

        # Preset shortcuts
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("📁 Use Built-in Demo Repo ('sample_repo')", use_container_width=True):
                st.session_state["folder_path_input"] = "sample_repo"
                st.rerun()
        with btn_col2:
            if st.button("📍 Use Current Project Folder ('.')", use_container_width=True):
                st.session_state["folder_path_input"] = "."
                st.rerun()

        folder_path = st.text_input(
            "Enter local repository folder path:",
            value=st.session_state.get("folder_path_input", "sample_repo"),
            placeholder="e.g. sample_repo, ./my-project, /Users/username/projects/my-repo",
            help="Provide absolute or relative directory path to the repository root.",
        )

        # Live directory inspection
        if folder_path.strip():
            summary = scan_directory_summary(folder_path.strip())
            if summary["exists"]:
                st.success(
                    f"✅ Found **{summary['file_count']}** supported code/doc files in `{folder_path}` "
                    f"(ignored `node_modules`, `.git`, `.venv`, `.env` automatically)."
                )
                with st.expander("👀 Preview detected files in folder", expanded=False):
                    for f in summary["supported_files"][:30]:
                        st.markdown(f"- `{f}`")
                    if len(summary["supported_files"]) > 30:
                        st.caption(f"... and {len(summary['supported_files']) - 30} more files.")
            else:
                st.error(f"❌ Directory path `{folder_path}` does not exist on disk.")

        if st.button("🚀 Index Repository Folder", type="primary", use_container_width=True):
            if not folder_path.strip() or not os.path.isdir(folder_path.strip()):
                st.error(f"Directory path '{folder_path}' does not exist or is not a folder.")
            else:
                with st.spinner(f"Scanning and indexing folder '{folder_path}' into FAISS..."):
                    try:
                        stats = pipeline.index_from_directory(
                            folder_path.strip(),
                            chunk_size=settings["chunk_size"],
                            chunk_overlap=settings["chunk_overlap"],
                        )
                        st.success(
                            f"🎉 Repository indexed successfully! "
                            f"{stats['files_indexed']} files converted to {stats['chunks_created']} semantic chunks."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during folder indexing: {e}")

    # --- TAB 2: UPLOAD ZIP ARCHIVE ---
    with tab_zip:
        st.markdown("Upload a downloaded repository archive (e.g. GitHub 'Download ZIP' or your zipped project folder).")
        uploaded_zip = st.file_uploader(
            "Upload repository .zip archive",
            type=["zip"],
            help="Upload a .zip file containing the entire repository tree.",
        )

        if st.button("🚀 Index ZIP Repository", type="primary", key="index_zip_btn"):
            if uploaded_zip is None:
                st.warning("Please upload a .zip file first.")
            else:
                with st.spinner("Extracting repository tree and building FAISS vector store..."):
                    try:
                        stats = pipeline.index_from_zip(
                            uploaded_zip,
                            chunk_size=settings["chunk_size"],
                            chunk_overlap=settings["chunk_overlap"],
                        )
                        st.success(
                            f"🎉 ZIP repository indexed successfully! "
                            f"{stats['files_indexed']} files converted to {stats['chunks_created']} chunks."
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to index ZIP archive: {e}")

    # --- TAB 3: UPLOAD LOOSE FILES ---
    with tab_files:
        st.markdown("Select individual source/documentation files manually.")
        uploaded_files = st.file_uploader(
            "Select individual files to index",
            accept_multiple_files=True,
            type=["py", "js", "jsx", "ts", "tsx", "html", "css", "md", "txt", "json", "java", "cpp", "c", "sql"],
        )

        if st.button("🚀 Index Selected Files", type="primary", key="index_files_btn"):
            if not uploaded_files:
                st.warning("Please upload repository files first.")
            else:
                with st.spinner("Processing files and building FAISS index..."):
                    docs = load_from_uploaded_files(uploaded_files)
                    if not docs:
                        st.error("No valid supported files found in upload.")
                    else:
                        try:
                            stats = pipeline.index_documents(
                                docs,
                                chunk_size=settings["chunk_size"],
                                chunk_overlap=settings["chunk_overlap"],
                            )
                            st.success(
                                f"🎉 Indexed {stats['files_indexed']} files into {stats['chunks_created']} chunks!"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to index files: {e}")

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

    current_query_value = st.session_state.get("selected_query", "")
    user_query = st.text_input(
        "Enter your question about the repository:",
        value=current_query_value,
        placeholder="e.g., How is authentication implemented? Which file handles database connection?",
    )

    ask_btn = st.button("🤖 Ask AI", type="primary")

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
    if st.session_state.get("last_result"):
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
