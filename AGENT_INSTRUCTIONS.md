# Project Documentation: Chunk-Tree RAG Pipeline

This document is designed for new developers or AI agents to quickly understand the project architecture, operational flow, and existing vulnerabilities/technical debt.

## 📌 Project Overview
The project is a Retrieval-Augmented Generation (RAG) pipeline built around a **"Chunk-Tree" architecture**. Instead of directly querying all document chunks, it uses a hierarchical approach to filter down the context:
1. **Root level**: Document-level summaries.
2. **Parent level**: Section-level summaries grouping multiple chunks.
3. **Child level**: The raw text chunks extracted from the PDFs.

The pipeline pulls files from Microsoft Drive, processes PDFs into the chunk-tree structure using **Gemini** (Google), stores vector embeddings in ChromaDB, and runs a multi-phase QA pipeline on questions provided in an Excel sheet. All AI reasoning and generation in this project uses **Gemini** only.

---

## 🏗️ Architecture & Component Flow

1. **`fetch_drive.py` (Data Ingestion - Drive)**
   - Authenticates with Microsoft Graph API using MSAL (Client Credentials flow).
   - Downloads new `.pdf` and `.xlsx` files from a specified Drive folder into `data/input`.

2. **`ingest_pdfs.py` (Data Ingestion - Local)**
   - Reads PDFs from `data/input` using `pdfplumber`.
   - **Chunking Strategy**: Splits text fundamentally by double newlines (`\n\n`). 
   - Outputs a flat list of text dictionaries to `cache/raw_chunks.json`.

3. **`build_chunk_tree.py` (Tree Generation)**
   - Uses Gemini (`gemini-2.5-flash`) to build the hierarchy.
   - Groups raw chunks by `CHUNK_GROUP_SIZE` (default: 5) and generates a **Parent Summary**.
   - Groups all Parent Summaries and generates a **Root Summary** for the entire document.
   - Saves the structured hierarchy as JSON files in `cache/chunk_trees/`.

4. **`setup_vector_db.py` (Database Initialization)**
   - Initializes a Persistent ChromaDB at `vector_db/`.
   - Populates three collections: `root_summaries`, `parent_summaries`, and `child_chunks` (all used by the retriever).

5. **`chunk_tree_retriever.py` (The RAG Engine)**
   Answers questions using **Gemini** via a multi-phase process:
   - **Phase 1 (Filter)**: Queries ChromaDB `root_summaries` to find the top relevant PDFs.
   - **Phase 2 (Reason)**: Uses vector search over `parent_summaries` (or Gemini over parent summaries if needed) to select relevant parent sections.
   - **Phase 3 (Extract)**: Retrieves and ranks child chunks; merges with direct chunk search for better recall.
   - **Phase 4 (Generate)**: Gemini (`gemini-2.5-pro`) generates the final answer from the selected chunks. A fallback pass uses a broader chunk search if the first attempt returns no answer.

6. **`run_qa_pipeline.py` (Batch Execution)**
   - Automatically finds `.xlsx` files in `data/input`.
   - Reads questions from the Excel file, runs the retriever for the first 5 questions (for rapid testing), and outputs a new Excel file with the generated answers.

---

## ⚠️ Vulnerabilities, Security Risks & Technical Debt

### 1. Security & Credentials
- **Environment Variables**: The project relies on `.env` for `MS_CLIENT_SECRET`, `GEMINI_API_KEY`, etc. Ensure `.env` is included in `.gitignore` so secrets are not accidentally committed to source control.
- **In-Memory Secrets**: Handled adequately via `dotenv`, but fallbacks in `config.py` (e.g. `"your_gemini_api_key"`) might cause silent execution failures if keys are missing but not explicitly validated.

### 2. Scalability & Technical Debt
- **Weak PDF Chunking (`ingest_pdfs.py`)**: The current strategy uses a simple `\n\n` split. This is unreliable for complex PDFs, multi-column layouts, or tables. 
  - *Recommendation*: Upgrade to `LangChain`'s `RecursiveCharacterTextSplitter` or a semantic chunker.
- **Hardcoded Limit Constraints**:
  - `run_qa_pipeline.py` is hardcoded to `.head(5)` to only evaluate 5 questions by default.
- **JSON Memory Overhead (`build_chunk_tree.py`)**: Building chunk trees loads the entire `raw_chunks.json` into memory. For large numbers of PDFs, this will cause Out-Of-Memory (OOM) errors.
  
### 3. Error Handling
- **Silent Failures**: Functions like `generate_summary` catch exceptions and return an error string instead of failing explicitly. This can pollute the vector database with string literals like `"Error: Could not generate summary...""` without the system crashing, causing weird retrieval behaviors down the line.

---

## 🚀 How to Act / Future Agent Roadmap
When assigned to work on this repository, an Agent should prioritize the following refactoring:
1. Implement `RecursiveCharacterTextSplitter` in `ingest_pdfs.py`.
2. Add robust validation logic prior to embedding inserts (avoid putting "Error strings" into Chroma).
3. (Done) `parent_summaries` is populated in ChromaDB; Phase 2 uses vector search over parents, with Gemini fallback when needed.
