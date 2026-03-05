# Chunk-Tree RAG Pipeline

This repo contains a Python Retrieval-Augmented Generation (RAG) pipeline built around a hierarchical **chunk-tree** representation of PDFs, with ChromaDB as the vector store and Google Gemini for summarization and answering questions.

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

Create a `.env` file in the project root:

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
   python run_qa_pipeline.py
   ```

## Resetting local state (start fresh locally)

To wipe local caches and the vector DB (without touching git-tracked code):

```bash
python ingest_pdfs.py --clear-cache
```

Then re-run the pipeline steps above.

