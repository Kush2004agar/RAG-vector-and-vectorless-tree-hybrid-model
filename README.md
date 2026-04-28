# RAG V3 (Single Qdrant Pipeline)

This repository now implements a clean V3 Retrieval-Augmented Generation (RAG) architecture with **Qdrant as the only retrieval backend**.

## V3 retrieval flow

`query -> process -> filter -> Qdrant search -> rerank -> context -> LLM`

There is exactly one retriever path, one vector search call per query, and no route-switching or graph expansion.

## Project structure

```text
rag_v3/
  ingestion/
    chunker.py
    feature_extractor.py
  indexing/
    embedder.py
    qdrant_client.py
    index_builder.py
  retrieval/
    query_processor.py
    filter_builder.py
    retriever.py
    schemas.py
  ranking/
    reranker.py
  context/
    context_builder.py
  serving/
    pipeline.py
```

## Data models

- `Query`: `raw`, `cleaned`, `tokens`, `feature_ids`
- `Candidate`: `id`, `text`, `score`, `metadata`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Qdrant locally with Docker

1. Start Qdrant:
   ```bash
   ./scripts/qdrant_local.sh up
   ```
2. Verify status:
   ```bash
   ./scripts/qdrant_local.sh status
   ```
3. Ensure your `.env` contains:
   ```bash
   QDRANT_URL=http://localhost:6333
   # QDRANT_API_KEY=...   # only if you enable auth
   ```
4. Stop when done:
   ```bash
   ./scripts/qdrant_local.sh down
   ```

Create `.env` with at least:

```bash
GEMINI_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=... # optional
```

## Usage

1. Put PDFs into `data/input/`.
2. Extract chunks:
   ```bash
   python ingest_pdfs.py
   ```
3. Build Qdrant `documents` collection:
   ```bash
   python setup_vector_db.py
   ```
4. Ask one question:
   ```bash
   python -c "from rag_v3.serving.pipeline import RagV3Pipeline; print(RagV3Pipeline().answer('your question'))"
   ```
5. Or batch from Excel:
   ```bash
   python run_qa_pipeline.py
   ```


## Evaluation

Use `rag_v3/evaluation` modules to compute retrieval quality (Recall@K and MRR) from JSON datasets with `query` and `relevant_chunk_ids`.
