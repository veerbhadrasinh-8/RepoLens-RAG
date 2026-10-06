# RepoLens --- Master Build Prompt

## AI Codebase RAG Assistant

> **Purpose:** Give this entire document to an AI coding assistant and
> ask it to build the project step-by-step. The goal is a small but
> interview-worthy RAG project that can be built quickly, understood
> easily, and demonstrated reliably.

------------------------------------------------------------------------

# 1. PROJECT OVERVIEW

Build a project called **RepoLens**.

RepoLens is a **Retrieval-Augmented Generation (RAG) assistant for
software repositories**.

The user provides a small software repository by uploading supported
source/documentation files. RepoLens indexes the repository, stores
vector embeddings in a local vector database, and allows the user to ask
questions about the codebase.

Example questions:

-   "What does this project do?"
-   "How is authentication implemented?"
-   "Which file handles database connection?"
-   "Explain the API flow."
-   "Where is JWT verification performed?"
-   "What happens when a user logs in?"
-   "Which files are related to authentication?"

The system must retrieve only the most relevant pieces of the repository
and provide them to an LLM. The LLM then generates the final
natural-language answer.

The project should visibly demonstrate the complete RAG pipeline:

**Documents → Chunking → Embeddings → Vector Store → Retrieval → Context
→ LLM → Answer + Sources**

------------------------------------------------------------------------

# 2. IMPORTANT CONCEPTUAL RULE

Do NOT confuse RAG with the LLM.

The project must follow this conceptual separation:

### RAG / Retrieval Layer

Responsible for:

-   loading repository content
-   cleaning/extracting text
-   splitting text into chunks
-   creating embeddings
-   storing embeddings
-   embedding the user's question
-   performing similarity search
-   returning the most relevant chunks

### LLM Layer

Responsible for:

-   understanding the retrieved chunks
-   combining information from multiple chunks
-   answering the user's question
-   explaining code
-   summarizing
-   following response instructions
-   refusing to invent information when the context does not contain the
    answer

The retriever should NOT generate the final answer.

The LLM should NOT be expected to search the entire repository itself.

------------------------------------------------------------------------

# 3. CORE ARCHITECTURE

Use this architecture:

``` text
                         ┌─────────────────────┐
                         │      User           │
                         │ Upload Repository   │
                         │ Ask Question        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Frontend  │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              Indexing Flow                  Query Flow
                     │                             │
                     ▼                             ▼
              File Loader                  User Question
                     │                             │
                     ▼                             ▼
                Chunking                   Query Embedding
                     │                             │
                     ▼                             ▼
              Embeddings                    FAISS Search
                     │                             │
                     ▼                             ▼
                FAISS DB                  Top-K Chunks
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                          Retrieved Context
                                    │
                                    ▼
                              Prompt Builder
                                    │
                                    ▼
                              Gemini LLM
                                    │
                                    ▼
                         Final Answer + Sources
                                    │
                                    ▼
                              Streamlit UI
```

------------------------------------------------------------------------

# 4. TECHNOLOGY STACK

Use a simple, reliable stack.

## Frontend

-   Python
-   Streamlit

## RAG Framework

-   LangChain

Use LangChain where it genuinely simplifies the pipeline. Do not add
unnecessary abstractions.

## Embedding Model

Use a free/local Hugging Face sentence-transformers embedding model.

Recommended:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model should run locally.

## Vector Database

Use:

``` text
FAISS
```

Reason:

-   simple
-   local
-   fast
-   no separate database server
-   easy to explain in an interview

## LLM

Use Google's Gemini API.

Use a currently available Gemini model supported by the installed
Google/LangChain integration.

Store the API key in:

``` text
.env
```

Never hard-code the API key.

## Environment

Python virtual environment.

------------------------------------------------------------------------

# 5. PROJECT SCOPE

Keep the project intentionally small.

Do NOT add:

-   user authentication
-   payment
-   PostgreSQL
-   Redis
-   Docker unless specifically needed
-   microservices
-   deployment complexity
-   complicated frontend frameworks
-   unnecessary database tables
-   model fine-tuning

The objective is a clean RAG demonstration, not a production SaaS
platform.

------------------------------------------------------------------------

# 6. SUPPORTED FILE TYPES

Initially support:

``` text
.py
.js
.jsx
.ts
.tsx
.html
.css
.md
.txt
.json
```

Optionally support:

``` text
.java
.cpp
.c
.sql
```

Ignore:

``` text
node_modules/
.git/
__pycache__/
.venv/
venv/
dist/
build/
.env
```

Never upload or index secrets.

------------------------------------------------------------------------

# 7. INPUTS TO THE SYSTEM

There are two different inputs.

## Input A --- Knowledge Base

The repository files.

Example:

``` text
project/
├── README.md
├── backend/
│   ├── server.js
│   ├── auth.js
│   └── database.js
└── frontend/
    └── app.js
```

These files are processed during indexing.

## Input B --- User Query

Example:

``` text
How is authentication implemented?
```

The user query is used during retrieval.

------------------------------------------------------------------------

# 8. COMPLETE RAG PIPELINE

Implement the following exact pipeline.

## STEP 1 --- File Upload

User uploads files through Streamlit.

For a first version, allow multiple files rather than requiring GitHub
authentication.

Do NOT require a GitHub OAuth integration.

The user can select files from their local repository and upload them.

------------------------------------------------------------------------

## STEP 2 --- File Loading

Read the contents of supported files.

For every loaded file, preserve metadata:

``` text
source
file_type
file_path
```

Example:

``` python
metadata = {
    "source": "backend/auth.js",
    "file_path": "backend/auth.js",
    "file_type": "javascript"
}
```

Metadata is important because the UI needs to show sources later.

------------------------------------------------------------------------

# 9. STEP 3 --- CHUNKING

Do not send the entire repository to the LLM.

Split files into smaller chunks.

Recommended initial settings:

``` text
chunk_size = 800
chunk_overlap = 100
```

These are starting values, not sacred constants.

The implementation should make them configurable.

Each chunk should retain its original metadata.

Example:

``` text
Chunk:
"JWT token is generated after successful login..."

Metadata:
source = backend/auth.js
```

Explain in comments that chunking helps:

-   retrieval precision
-   context-window efficiency
-   lower token usage
-   better relevance

------------------------------------------------------------------------

# 10. STEP 4 --- EMBEDDINGS

Convert every chunk into a vector.

Example concept:

``` text
"JWT authentication"
        ↓
Embedding Model
        ↓
[0.21, -0.73, 0.44, 0.82, ...]
```

Use:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

Do not manually implement neural embeddings.

Use the embedding library through a clean abstraction.

------------------------------------------------------------------------

# 11. STEP 5 --- VECTOR STORE

Store embeddings in FAISS.

Conceptually:

``` text
Chunk 1 → Vector 1
Chunk 2 → Vector 2
Chunk 3 → Vector 3
...
Chunk 500 → Vector 500
```

FAISS allows the system to find vectors that are semantically close to
the user's question.

Persist the index locally so it can be reused when practical.

Example directory:

``` text
data/
└── faiss_index/
```

Also preserve the document metadata needed to identify the source.

------------------------------------------------------------------------

# 12. STEP 6 --- USER QUERY EMBEDDING

When the user asks:

``` text
How is authentication implemented?
```

Convert the question into an embedding using the SAME embedding model
used for the documents.

Important:

Use the same embedding space for documents and queries.

------------------------------------------------------------------------

# 13. STEP 7 --- SIMILARITY SEARCH

Search FAISS using the query vector.

Retrieve the top relevant chunks.

Recommended initial value:

``` text
top_k = 4
```

Make this configurable in the UI/sidebar if simple to implement.

The retrieval result should conceptually look like:

``` text
[
    {
        "content": "...",
        "source": "backend/auth.js",
        "score": 0.87
    },
    {
        "content": "...",
        "source": "backend/middleware.js",
        "score": 0.82
    }
]
```

The exact score format depends on the vector store implementation.

Do not claim a score is a probability.

------------------------------------------------------------------------

# 14. STEP 8 --- CONTEXT CONSTRUCTION

Take the retrieved chunks and construct the context given to the LLM.

Example:

``` text
SOURCE: backend/auth.js

JWT token is generated after successful login...

SOURCE: backend/middleware.js

The JWT token is verified before protected routes are accessed...
```

Do not include all repository chunks.

Only include retrieved chunks.

------------------------------------------------------------------------

# 15. STEP 9 --- PROMPT TO THE LLM

Use a prompt similar to this:

``` text
You are RepoLens, an AI assistant that answers questions about
a software repository.

Answer the user's question using ONLY the provided repository context.

Rules:
1. Do not invent code, files, functions, or behavior.
2. If the answer is not present in the context, clearly say that
   the repository context does not contain enough information.
3. Explain the answer clearly and concisely.
4. When possible, mention the relevant source file.
5. If multiple files are involved, explain how they relate.
6. Prefer evidence from the retrieved code over general assumptions.

REPOSITORY CONTEXT:

{retrieved_context}

USER QUESTION:

{question}
```

The actual prompt should be constructed programmatically.

Do not hard-code a fake answer.

------------------------------------------------------------------------

# 16. STEP 10 --- LLM GENERATION

Send:

``` text
system instructions
+
retrieved context
+
user question
```

to Gemini.

The LLM generates the final answer.

Example:

``` text
The application uses JWT-based authentication.

The login logic generates the JWT after successful
credential validation in `backend/auth.js`. The token
is then checked by middleware before protected routes
are accessed.

Sources:
- backend/auth.js
- backend/middleware.js
```

The LLM's responsibility is generation and reasoning over the supplied
context.

------------------------------------------------------------------------

# 17. STEP 11 --- SOURCE DISPLAY

Do NOT rely on the LLM to invent source names.

Sources should be generated from the metadata of the retrieved
documents.

Show:

``` text
Answer
────────────────────────

The application uses JWT authentication...

Sources
────────────────────────
1. backend/auth.js
2. backend/middleware.js
```

Optionally show:

``` text
Retrieved Chunks
────────────────────────
backend/auth.js
Similarity: ...

backend/middleware.js
Similarity: ...
```

The source list should come from retrieval metadata, not from
hallucinated LLM output.

------------------------------------------------------------------------

# 18. USER INTERFACE

Create a clean Streamlit interface.

Suggested layout:

``` text
┌─────────────────────────────────────────────┐
│ RepoLens                                    │
│ AI Codebase & Documentation Assistant       │
├─────────────────────────────────────────────┤
│                                             │
│ Upload repository files                     │
│ [ Upload Files ]                            │
│                                             │
│ [ Index Repository ]                        │
│                                             │
├─────────────────────────────────────────────┤
│ Repository Status                            │
│ Files: 12                                   │
│ Chunks: 84                                  │
│ Status: Indexed                             │
├─────────────────────────────────────────────┤
│ Ask RepoLens                                │
│                                             │
│ [ How is authentication implemented? ]      │
│                                             │
│ [ Ask AI ]                                  │
├─────────────────────────────────────────────┤
│ Answer                                      │
│                                             │
│ ...                                         │
│                                             │
├─────────────────────────────────────────────┤
│ Sources                                     │
│                                             │
│ backend/auth.js                             │
│ backend/middleware.js                       │
└─────────────────────────────────────────────┘
```

Add a sidebar with:

``` text
Settings

Top K: 4
Chunk Size: 800
Chunk Overlap: 100

Show Retrieved Chunks: ✓
```

Do not over-design the UI.

------------------------------------------------------------------------

# 19. RETRIEVAL INSPECTOR

This is an important interview/demo feature.

Provide an expandable section:

``` text
🔍 Retrieved Context
```

Inside it show:

``` text
1. backend/auth.js
   Score: ...

   JWT token is generated...

2. backend/middleware.js
   Score: ...

   JWT token is verified...
```

This allows the interviewer to see the RAG retrieval process.

It demonstrates that the application is not simply:

``` text
Question → LLM
```

It is:

``` text
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```

------------------------------------------------------------------------

# 20. PROJECT DIRECTORY STRUCTURE

Create a clean structure similar to:

``` text
repolens/
│
├── app.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── data/
│   └── faiss_index/
│
└── tests/
    └── test_basic.py
```

Keep the modules simple.

Do not create unnecessary abstractions.

------------------------------------------------------------------------

# 21. RESPONSIBILITY OF EACH FILE

## app.py

Responsible for:

-   Streamlit UI
-   file upload
-   index button
-   question input
-   displaying answer
-   displaying sources
-   displaying retrieved chunks

It should not contain all RAG logic.

------------------------------------------------------------------------

## loader.py

Responsible for:

-   reading supported files
-   filtering unsupported files
-   ignoring unwanted directories/files
-   attaching metadata

------------------------------------------------------------------------

## chunker.py

Responsible for:

-   splitting documents
-   maintaining metadata
-   configurable chunk size
-   configurable overlap

------------------------------------------------------------------------

## embeddings.py

Responsible for:

-   initializing embedding model
-   creating embeddings

------------------------------------------------------------------------

## vector_store.py

Responsible for:

-   creating FAISS index
-   adding documents
-   saving index
-   loading index

------------------------------------------------------------------------

## retriever.py

Responsible for:

-   receiving user query
-   converting query to embedding
-   searching FAISS
-   returning top-k relevant chunks

------------------------------------------------------------------------

## prompt.py

Responsible for:

-   constructing the LLM prompt
-   inserting retrieved context
-   inserting user question
-   enforcing anti-hallucination instructions

------------------------------------------------------------------------

## llm.py

Responsible for:

-   initializing Gemini
-   sending prompt
-   returning LLM response

------------------------------------------------------------------------

## rag_pipeline.py

Responsible for orchestrating:

``` text
Question
→ Retrieval
→ Context
→ Prompt
→ LLM
→ Final Answer
```

------------------------------------------------------------------------

# 22. ENVIRONMENT VARIABLES

Use:

``` text
GEMINI_API_KEY=your_key_here
```

Create:

``` text
.env.example
```

containing:

``` text
GEMINI_API_KEY=
```

Never commit `.env`.

Add to `.gitignore`:

``` text
.env
__pycache__/
.venv/
venv/
data/faiss_index/
*.pyc
.DS_Store
```

------------------------------------------------------------------------

# 23. REQUIREMENTS

Generate a minimal `requirements.txt`.

Use compatible current versions.

The exact package names should match the implementation.

Likely packages include:

``` text
streamlit
langchain
langchain-community
langchain-google-genai
sentence-transformers
faiss-cpu
python-dotenv
```

Do not add packages unless they are actually used.

Before finalizing, verify imports against the installed/current package
APIs.

------------------------------------------------------------------------

# 24. ERROR HANDLING

The application must handle:

### No files uploaded

Display:

``` text
Please upload repository files first.
```

### Index button clicked without files

Display a clear warning.

### Question asked before indexing

Display:

``` text
Please index the repository before asking questions.
```

### Missing API key

Display:

``` text
Gemini API key is not configured.
Please add GEMINI_API_KEY to .env.
```

### Unsupported file

Skip it safely.

### Empty file

Skip it safely.

### LLM/API failure

Show a user-friendly error without exposing secrets.

### No relevant retrieval

If the retriever returns weak/no useful context, tell the LLM to say
that the repository does not contain enough information.

Do not fabricate an answer.

------------------------------------------------------------------------

# 25. IMPORTANT RAG BEHAVIOR

The application should NOT send every chunk to Gemini.

Example:

``` text
Repository = 500 chunks

User asks:
"How does authentication work?"

Retriever:
Top 4 chunks

LLM receives:
4 chunks + question + instructions

NOT:
500 chunks + question
```

This is the core reason the RAG architecture exists.

Benefits:

-   less token usage
-   lower cost
-   faster responses
-   less irrelevant information
-   better grounding
-   works with larger repositories

------------------------------------------------------------------------

# 26. WHAT HAPPENS DURING INDEXING

Explain this in the UI/README:

``` text
Repository Files
      ↓
Load
      ↓
Extract Text
      ↓
Chunk
      ↓
Embed
      ↓
Store in FAISS
```

This happens when the user clicks:

``` text
Index Repository
```

------------------------------------------------------------------------

# 27. WHAT HAPPENS DURING A QUESTION

``` text
User Question
      ↓
Create Query Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Build Context
      ↓
Build Prompt
      ↓
Gemini
      ↓
Final Answer
      ↓
Show Sources
```

This happens every time the user clicks:

``` text
Ask AI
```

------------------------------------------------------------------------

# 28. RAG VS NORMAL LLM

The README should explain:

## Normal LLM

``` text
Question
   ↓
LLM
   ↓
Answer
```

The model relies primarily on its pretrained knowledge and the prompt.

## RAG

``` text
Question
   ↓
Retriever
   ↓
Relevant external knowledge
   ↓
LLM
   ↓
Grounded answer
```

RAG gives the LLM information that was not necessarily present in its
original training data.

------------------------------------------------------------------------

# 29. RAG VS FINE-TUNING

Be able to explain:

### RAG

Changes the information available to the model at inference time.

``` text
Knowledge → retrieved dynamically
```

### Fine-tuning

Changes model behavior/weights through additional training.

``` text
Model → trained further
```

For RepoLens, RAG is preferable because repositories can change
frequently and we want to retrieve current repository content without
retraining a model.

------------------------------------------------------------------------

# 30. WHY FAISS?

Answer:

> FAISS is a lightweight vector similarity search library that works
> locally and is easy to use for a small interview project. It avoids
> the operational complexity of running a separate vector database
> server.

Do not claim FAISS is always better than production vector databases.

------------------------------------------------------------------------

# 31. WHY EMBEDDINGS?

Embeddings allow semantic similarity.

For example:

``` text
Question:
"How do users log in?"

Relevant code:
"authenticateUser() validates email and password..."
```

The wording is different, but their meaning is related.

Keyword search may miss this.

Embedding similarity can retrieve it.

------------------------------------------------------------------------

# 32. WHY NOT SEND THE ENTIRE REPOSITORY TO THE LLM?

Answer:

> Sending the entire repository is inefficient and may exceed the
> model's context window. RAG retrieves only the information relevant to
> the current question, reducing tokens, cost, latency, and irrelevant
> context.

------------------------------------------------------------------------

# 33. WHAT EXACTLY DOES THE LLM DO?

The LLM receives:

``` text
System instructions
+
Retrieved repository context
+
User question
```

Example:

``` text
SYSTEM:
Answer using only the repository context.

CONTEXT:
auth.js:
JWT token is generated after login.

middleware.js:
JWT token is verified before protected routes.

QUESTION:
How does authentication work?
```

The LLM then generates:

``` text
Authentication uses JWT.

The login process generates a JWT token after
successful authentication. The middleware then
verifies the token before protected routes are accessed.
```

The LLM is therefore responsible for **language understanding,
synthesis, explanation, and generation**.

------------------------------------------------------------------------

# 34. WHAT EXACTLY DOES THE RAG SYSTEM DO?

RAG performs the retrieval pipeline:

``` text
Repository
→ chunks
→ embeddings
→ vector index

Question
→ query embedding
→ similarity search
→ relevant chunks
```

Its retrieval output is essentially:

``` text
Relevant chunks + metadata
```

It does not need to generate the final natural-language response.

------------------------------------------------------------------------

# 35. DEMO SCRIPT

For the interview, use a tiny repository with approximately:

``` text
README.md
backend/server.js
backend/auth.js
backend/middleware.js
backend/database.js
frontend/app.js
```

Ask these questions in order:

### Demo 1

``` text
What does this project do?
```

### Demo 2

``` text
How is authentication implemented?
```

### Demo 3

``` text
Which file handles database connection?
```

### Demo 4

Ask something not contained in the repository:

``` text
How does this project process credit card payments?
```

The system should respond that the retrieved repository context does not
contain enough information.

This demonstrates grounding and helps show that the system is designed
to reduce hallucination.

------------------------------------------------------------------------

# 36. INTERVIEW QUESTIONS TO PREPARE

The developer should understand these answers before presenting the
project.

## Basic

1.  What is RAG?
2.  Why did you use RAG?
3.  What is an embedding?
4.  Why do we need embeddings?
5.  What is FAISS?
6.  What is a vector database?
7.  What is similarity search?
8.  What is top-k retrieval?
9.  What is chunking?
10. Why do chunks have overlap?

## Architecture

11. What happens when the repository is uploaded?
12. What happens when the user asks a question?
13. Where is the question embedded?
14. Where are document embeddings stored?
15. What exactly is sent to Gemini?
16. Why don't you send the entire repository?
17. What does the retriever return?
18. What does the LLM return?

## Deeper

19. RAG vs fine-tuning?
20. What happens if retrieval returns irrelevant chunks?
21. How would you improve retrieval?
22. What is a context window?
23. What causes hallucinations?
24. How does RAG reduce hallucination?
25. Why use the same embedding model for documents and queries?
26. How would this scale to a huge repository?
27. Why might you replace FAISS in production?
28. How would you add GitHub integration?
29. How would you protect secrets?
30. How would you evaluate retrieval quality?

------------------------------------------------------------------------

# 37. POSSIBLE FUTURE IMPROVEMENTS

Do not implement these unless time permits.

Possible improvements:

``` text
GitHub URL ingestion
        ↓
Automatic repository cloning
        ↓
Automatic indexing
```

Other improvements:

-   hybrid keyword + semantic search
-   reranking
-   code-aware chunking
-   AST-based parsing
-   repository tree visualization
-   conversation history
-   streaming answers
-   multiple vector indexes
-   PostgreSQL + pgvector
-   Chroma/Qdrant/Pinecone
-   authentication
-   deployment

Mention these as future improvements rather than building them now.

------------------------------------------------------------------------

# 38. DEVELOPMENT ORDER

The AI coding assistant MUST build the project in this order.

## Phase 1 --- Project Setup

Create:

``` text
app.py
src/
requirements.txt
.env.example
.gitignore
README.md
```

Create virtual environment instructions.

Verify Python and pip.

------------------------------------------------------------------------

## Phase 2 --- File Loader

Implement file upload and loading.

Test with:

``` text
README.md
auth.js
database.js
```

Verify that text and metadata are correct.

Do not continue until this works.

------------------------------------------------------------------------

## Phase 3 --- Chunking

Implement chunking.

Print/display:

``` text
Files loaded: X
Chunks created: Y
```

Verify metadata survives chunking.

------------------------------------------------------------------------

## Phase 4 --- Embeddings

Initialize the local embedding model.

Test one chunk.

Verify an embedding vector is generated.

------------------------------------------------------------------------

## Phase 5 --- FAISS

Create the FAISS vector store.

Insert all chunks.

Test a sample similarity search.

------------------------------------------------------------------------

## Phase 6 --- Retrieval

Implement:

``` text
query
→ embedding
→ FAISS
→ top-k chunks
```

Display retrieved sources.

Do not call Gemini yet.

Verify retrieval quality first.

------------------------------------------------------------------------

## Phase 7 --- Prompt

Create the prompt template.

Test that retrieved context and user question are inserted correctly.

------------------------------------------------------------------------

## Phase 8 --- Gemini

Connect Gemini.

Send:

``` text
instructions
+
context
+
question
```

Display the answer.

------------------------------------------------------------------------

## Phase 9 --- Sources

Show sources based on retrieved metadata.

Do not rely on LLM-generated source names.

------------------------------------------------------------------------

## Phase 10 --- Retrieval Inspector

Add expandable retrieved-context UI.

------------------------------------------------------------------------

## Phase 11 --- Error Handling

Test:

-   empty upload
-   unsupported files
-   no index
-   no API key
-   API failure
-   empty question
-   no useful retrieval

------------------------------------------------------------------------

## Phase 12 --- Final Testing

Run at least:

``` text
"What does this project do?"

"How does authentication work?"

"Where is the database connection?"

"What payment gateway does this project use?"
```

The last question should produce a grounded "not found" style response
if payment information is absent.

------------------------------------------------------------------------

# 39. QUALITY REQUIREMENTS

The final application must:

-   run without import errors
-   have no hard-coded API key
-   not expose secrets
-   preserve source metadata
-   not send the whole repository to the LLM
-   retrieve top-k relevant chunks
-   provide grounded answers
-   display sources
-   handle errors gracefully
-   have a clean Streamlit interface
-   have a readable README
-   use modular Python files
-   be easy to explain in an interview

------------------------------------------------------------------------

# 40. AI CODING ASSISTANT RULES

When implementing this project:

1.  Do not generate the entire project blindly in one step.
2.  Build one phase at a time.
3.  After every phase, verify imports and functionality.
4.  Do not proceed when the current phase has errors.
5.  Use the current package APIs rather than outdated LangChain APIs.
6.  Keep dependencies minimal.
7.  Do not invent deprecated classes or functions.
8.  If a library API has changed, use the current supported API.
9.  Keep the code beginner-readable.
10. Add short comments explaining important RAG concepts.
11. Do not hide all logic inside one giant `app.py`.
12. Do not introduce unnecessary design patterns.
13. Do not add features outside the defined scope.
14. Never hard-code secrets.
15. Do not fabricate retrieval scores or source files.
16. Make source metadata traceable from uploaded file → chunk →
    retrieval → UI.
17. Test every stage independently before integrating the next stage.
18. If an implementation choice changes because of a package version,
    document the change in README.

------------------------------------------------------------------------

# 41. FINAL ACCEPTANCE TEST

The project is considered complete only when this works:

``` text
1. Start Streamlit
        ↓
2. Upload repository files
        ↓
3. Click "Index Repository"
        ↓
4. Files are loaded
        ↓
5. Files are chunked
        ↓
6. Embeddings are created
        ↓
7. FAISS index is created
        ↓
8. User asks a question
        ↓
9. Question is embedded
        ↓
10. FAISS retrieves top-k chunks
        ↓
11. Retrieved context is inserted into prompt
        ↓
12. Gemini generates answer
        ↓
13. Answer is displayed
        ↓
14. Source files are displayed
        ↓
15. Retrieved chunks can be inspected
```

------------------------------------------------------------------------

# 42. ONE-SENTENCE INTERVIEW EXPLANATION

Memorize this:

> **"RepoLens is a RAG-based codebase assistant that converts repository
> files into embeddings, stores them in FAISS, retrieves the most
> relevant code for a user's question, and gives that retrieved context
> to Gemini so it can generate a grounded answer with source
> references."**

------------------------------------------------------------------------

# 43. 30-SECOND ARCHITECTURE EXPLANATION

If the interviewer asks:

**"Explain your architecture."**

Say:

> "The application has an indexing pipeline and a query pipeline. During
> indexing, uploaded repository files are loaded, split into chunks,
> converted into embeddings using a sentence-transformer model, and
> stored in FAISS with their file metadata. When the user asks a
> question, the question is embedded using the same embedding model and
> used to perform similarity search in FAISS. The top relevant chunks
> are then placed into a prompt along with the question and sent to
> Gemini. Gemini generates the final answer, while the source metadata
> from retrieval is shown separately in the UI."

------------------------------------------------------------------------

# 44. MOST IMPORTANT MENTAL MODEL

Remember this:

``` text
                REPOSITORY
                     │
                     ▼
              ┌─────────────┐
              │   CHUNKS    │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ EMBEDDINGS  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │    FAISS    │
              └──────┬──────┘
                     │
                     │
USER QUESTION ───────┤
                     ▼
              ┌─────────────┐
              │ RETRIEVER   │
              └──────┬──────┘
                     │
               Top-K chunks
                     │
                     ▼
              ┌─────────────┐
              │    PROMPT   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   GEMINI    │
              │     LLM     │
              └──────┬──────┘
                     │
                     ▼
               FINAL ANSWER
```

The single most important distinction:

``` text
RAG → finds the information.

LLM → understands that information and generates the answer.
```

Do not describe RAG as the model that answers the user.

------------------------------------------------------------------------

# 45. FINAL INSTRUCTION TO THE CODING AI

Now build RepoLens according to this specification.

Start with **Phase 1 only**.

After completing Phase 1:

1.  show the created directory structure
2.  show the exact commands to install dependencies
3.  show the files created
4.  verify that the project starts
5.  identify any errors
6.  stop and wait for approval before moving to Phase 2

Do not skip phases.

Do not assume that an untested component works.

The priority is:

**Correctness → Simplicity → Explainability → Demo quality**

not unnecessary feature count.
