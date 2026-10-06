# RepoLens — AI Codebase & Documentation RAG Assistant

RepoLens is a lightweight, interview-worthy **Retrieval-Augmented Generation (RAG)** assistant built for software repositories. It allows developers to upload codebase files, indexes them into a local vector store using semantic embeddings, and answers complex architectural and implementation questions grounded strictly in the repository's actual code.

---

## 1. Quick Summary (Interview Pitch)

> **"RepoLens is a RAG-based codebase assistant that converts repository files into embeddings, stores them in FAISS, retrieves the most relevant code for a user's question, and gives that retrieved context to Gemini so it can generate a grounded answer with source references."**

---

## 2. Core Architecture & Pipeline

```text
               REPOSITORY FILES (Knowledge Base)
                             │
                             ▼
                    ┌─────────────────┐
                    │   File Loader   │  (Filters supported files, preserves metadata)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Chunker     │  (Recursive character splitting + overlap)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model │  (sentence-transformers/all-MiniLM-L6-v2)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  FAISS DB Index │  (Local similarity index)
                    └────────┬────────┘
                             │
                             │ (Similarity Search)
USER QUESTION ───────────────┤
                             ▼
                    ┌─────────────────┐
                    │    Retriever    │  (Top-K most relevant chunks)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Prompt Builder  │  (Grounding instructions + retrieved context)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Gemini LLM    │  (Reasoning & Answer Synthesis)
                    └────────┬────────┘
                             │
                             ▼
                    FINAL ANSWER + SOURCES
```

### The Core Separation: RAG vs LLM
- **RAG Layer (Retrieval):** Responsible for file ingestion, text chunking, embedding generation, FAISS indexing, query embedding, and returning top-K relevant chunks with metadata. It does **not** generate natural-language answers.
- **LLM Layer (Gemini):** Responsible for reading the retrieved snippets, synthesizing explanations across multiple files, adhering to strict grounding constraints, and refusing to invent non-existent code. It does **not** search the repository directly.

---

## 3. Technology Stack

- **UI / Frontend:** Streamlit (Python)
- **RAG Framework:** LangChain / LangChain Community
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Local, fast, no external API cost)
- **Vector Database:** FAISS CPU (Local, memory-efficient, no external server needed)
- **Large Language Model:** Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`)
- **Environment Management:** Python virtual environment (`.venv`) & `python-dotenv`

---

## 4. Why These Architectural Decisions?

| Decision | Why? |
| :--- | :--- |
| **Why RAG instead of standard LLM?** | A standard LLM relies only on pre-training data and cannot see private, proprietary, or newly updated code. RAG dynamically retrieves the exact source files at query time, guaranteeing grounded, hallucination-free answers. |
| **Why RAG instead of Fine-Tuning?** | Fine-tuning changes model weights through expensive training, yet codebases change constantly. RAG updates instantaneously just by rebuilding or updating the index—no GPU retraining needed. |
| **Why FAISS?** | FAISS is an efficient in-process vector similarity search library. It avoids running Docker or external database services (like Milvus/Pinecone), making it ideal for self-contained, reproducible demos. |
| **Why Embeddings instead of Keyword Search?** | Embeddings map semantic meaning into vector space. A question like *"How do users log in?"* accurately retrieves code like `authenticateUser(email, pass)` even if the word "login" never appears. |
| **Why Chunking with Overlap?** | LLM context windows and attention mechanisms degrade with massive raw text. Chunking isolates cohesive logic blocks, and overlap prevents splitting code across critical function boundaries. |

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Google Gemini API Key ([Get one at Google AI Studio](https://aistudio.google.com/))

### Step 1: Clone and Navigate to Directory
```bash
cd /path/to/RepoLens
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...your_actual_gemini_api_key...
```

---

## 6. Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 7. Demo Script (Interview Walkthrough)

To demonstrate the full power and reliability of RepoLens:

1. **Ingest Codebase (3 Options):**
   - **Option A (Local Folder - Recommended):** Enter the local path to any repository (or click `"📁 Use Built-in Demo Repo ('sample_repo')"`), preview detected files, and click `"🚀 Index Repository Folder"`.
   - **Option B (Repository ZIP):** Upload any project `.zip` archive (e.g. downloaded from GitHub) and click `"🚀 Index ZIP Repository"`.
   - **Option C (Loose Files):** Select individual files from disk.
2. **Indexing:** Watch RepoLens crawl directory trees, filter secret/ignored files (`.git`, `node_modules`, `.env`), chunk code, compute embeddings, and build the FAISS index.
3. **Ask Question 1 (Overview):**
   > *"What does this project do?"*
   - Observe the answer synthesized from `README.md`.
4. **Ask Question 2 (Feature Implementation):**
   > *"How is authentication implemented?"*
   - Observe the grounded answer citing `backend/auth.js` and `backend/middleware.js`.
5. **Ask Question 3 (Architecture/Components):**
   > *"Which file handles database connection?"*
   - Observe pinpoint retrieval of `backend/database.js`.
6. **Ask Question 4 (Anti-Hallucination Test):**
   > *"How does this project process credit card payments with Stripe?"*
   - Observe RepoLens refusing to invent facts and stating that the repository context lacks payment processing details.
7. **Inspect Retrieval:** Open the **🔍 Retrieved Context** accordion to show the interviewer the exact chunk texts and similarity scores retrieved from FAISS before the LLM was prompted.

---

## 8. Running Automated Tests

Run the test suite with pytest:
```bash
pytest tests/ -v
```
