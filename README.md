# Chunk-Tree RAG Pipeline

This repo contains a Python Retrieval-Augmented Generation (RAG) pipeline built around a hierarchical **chunk-tree** representation of PDFs. All summarization and question-answering is done by **Google Gemini** (Gemini); ChromaDB is used as the vector store.

The repo is set up so that a new user starts with **code only**: no cached trees, vector DB, or local data are tracked in git.

## Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- Google Gemini API key
- (Optional) Microsoft Graph / OneDrive app registration if you want to sync PDFs from the cloud

## Setup

```bash
git clone <this-repo>
cd rag+page\ index  # or your chosen folder name

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (you can copy from `.env.example`):

```bash
GEMINI_API_KEY=your_gemini_api_key
MS_CLIENT_ID=your_client_id          # optional, for OneDrive sync
MS_CLIENT_SECRET=your_client_secret  # optional
MS_TENANT_ID=your_tenant_id          # optional
MS_DRIVE_ID=your_drive_id            # optional
MS_FOLDER_ID=your_folder_id          # optional
```

## Typical workflow (fresh clone)

1. **(Optional) Sync PDFs and Excel files from OneDrive**

   ```bash
   python fetch_drive.py
   ```

   Or manually drop your `.pdf` and `.xlsx` files into `data/input/`.

2. **Ingest PDFs into raw chunks**

   ```bash
   python ingest_pdfs.py
   ```

3. **Build chunk trees (root, parents, children)**

   ```bash
   python build_chunk_tree.py
   ```

4. **Initialize and populate the vector DB**

   ```bash
   python setup_vector_db.py
   ```

5. **Ask a single question**

   ```bash
   python chunk_tree_retriever.py "Your question here"
   ```

6. **Run the batch QA pipeline from an Excel file**

   ```bash
   python run_qa_pipeline.py -i "Network Security and 5G Questions.xlsx"
   ```

   - Use `-i/--input` to explicitly target the questions file (recommended).
   - If you omit `-i`, the script auto-picks the first `.xlsx` in `data/input/` that is **not** named `answered_*.xlsx`.
   - By default this processes **all questions** in the Excel and saves progress every **50** (see `config.py`: `MAX_QUESTIONS`, `SAVE_PROGRESS_EVERY_N`).
   - To limit to e.g. 100: `python run_qa_pipeline.py -i "Network Security and 5G Questions.xlsx" -n 100`.

## Scaling (1000+ PDFs, 500+ questions)

- **Many PDFs**: In `config.py`, increase retrieval so the right docs and chunks are considered:
  - `ROOT_RETRIEVER_TOP_K` (default 12) — number of PDFs to consider per question; raise to 15–20 for 1000+ docs.
  - `INITIAL_CONTEXT_CHUNK_COUNT` (default 50) and `FALLBACK_CONTEXT_CHUNK_COUNT` (default 80) — chunks sent to Gemini; raise for better recall (e.g. 80 / 120).
  - `MAX_PARENT_CANDIDATES` (default 15) — section groups per question; raise if needed.
- You can override via environment variables, e.g. `ROOT_RETRIEVER_TOP_K=20 INITIAL_CONTEXT_CHUNK_COUNT=80 python run_qa_pipeline.py`.
- **Many questions**: By default all rows are processed. Set `MAX_QUESTIONS=500` in `.env` to cap. Use `-n 100` to process only the first 100. Progress is saved every `SAVE_PROGRESS_EVERY_N` questions (default 50) so a crash does not lose all work.

## Targeting a specific questions file (without editing `.env`)

- **Recommended**: pass the input file on the command line:

```bash
python run_qa_pipeline.py -i "Network Security and 5G Questions.xlsx"
```

- **Alternative**: set `DEFAULT_QUESTIONS_EXCEL` in `config.py` (kept in source control) if you want a repo-default input for your team. If set, it can be a filename in `data/input/` or a full path.

## Resetting local state (start fresh locally)

To wipe local caches and the vector DB (without touching git-tracked code):

```bash
python ingest_pdfs.py --clear-cache
```

Then re-run the pipeline steps above.

