# Agent instructions for this repo

These guidelines are for AI coding agents (and future contributors) working in this repository.

## Project purpose

This project implements a **hybrid hierarchical RAG** system over PDF documents:

- PDFs are ingested into paragraph-level chunks.
- Chunks are grouped into a **chunk tree** (root summary → parent/section summaries → child chunks).
- A hybrid retriever combines:
  - vector search (ChromaDB),
  - lexical scoring,
  - and tree-structured expansion
to build context for Google Gemini, which generates answers.

## Key modules and responsibilities

- `config.py`
  - Defines directories (`data/input`, `cache`, `vector_db`, `cache/chunk_trees`) and retrieval hyperparameters (e.g. `ROOT_RETRIEVER_TOP_K`, `INITIAL_CONTEXT_CHUNK_COUNT`).

- `ingest_pdfs.py`
  - Reads PDFs from `data/input/`.
  - Produces per-page paragraph chunks and stores them in `cache/raw_chunks.json`.
  - Handles cache-clearing via `--clear-cache`.

- `build_chunk_tree.py`
  - Groups chunks into **parent nodes** (sections) and generates:
    - parent summaries for each group,
    - a root summary per document,
    - and saves a chunk tree JSON per PDF (under `cache/chunk_trees/`).

- `setup_vector_db.py`
  - Builds ChromaDB indices:
    - `root_summaries` (document-level),
    - `parent_summaries` (section-level),
    - `child_chunks` (leaf chunks).

- `node_model.py`
  - Defines the unified `Node` dataclass used across the system (document, section, paragraph, summary).
  - Provides `legacy_tree_to_nodes(...)` to adapt the existing tree JSON format into an in-memory `node_id → Node` graph without changing the files on disk.

- `query_classifier.py`
  - Provides a lightweight rule-based classifier:
    - `classify_query(question) -> QueryType`
    - Types include: definition, comparison, reasoning, summarization, navigation, other.
  - Used to log intent and, over time, steer retrieval strategies.

- `chunk_tree_retriever.py`
  - Core retrieval & answering logic:
    - `load_node_graph(pdf_name)` builds a `Node` graph from a legacy chunk tree.
    - `multi_stage_retrieve(question)`:
      1. Retrieves likely PDFs via `root_summaries`.
      2. Retrieves parent summaries and their child chunks via the node graph and Chroma.
      3. Merges direct and global chunk candidates.
      4. Runs a hybrid reranking step (currently the existing lexical+distance score, factored for future weighting).
      5. Calls `expand_tree_neighbors(...)` to add parent/sibling context around top chunks.
    - `optimize_chunks_for_context(...)` deduplicates chunks before the final Gemini prompt.
    - `answer_question(question)` orchestrates:
      - query classification,
      - multi-stage retrieval,
      - context optimization,
      - Gemini call for the final answer.

- `run_qa_pipeline.py`
  - Batch pipeline that:
    - Reads questions from an Excel file in `data/input/`,
    - Calls `answer_question` for each,
    - Writes answers to `answered_<input>.xlsx`.

## Design and implementation guidelines for agents

- **Preserve the unified node model**
  - Prefer operating on `Node` instances and node graphs rather than raw dicts when working with tree structures.
  - When modifying tree-related logic, use `legacy_tree_to_nodes(...)` and keep the on-disk JSON format backward-compatible unless explicitly migrating it.

- **Keep retrieval stages explicit**
  - Maintain the structure:
    1. Query understanding (`classify_query`).
    2. Root selection.
    3. Parent + chunk retrieval.
    4. Tree expansion (`expand_tree_neighbors`).
    5. Hybrid reranking (`hybrid_rerank_candidates`).
    6. Context optimization (`optimize_chunks_for_context`).
    7. Answer generation (`generate_answer`).
  - When adding new signals (e.g. structural scores, keyword scores), plug them into the explicit stages instead of creating new ad-hoc code paths.

- **Be careful with vector DB changes**
  - If you alter collection schemas (e.g. adding `node_id` to metadata), also:
    - Update the code that writes to Chroma (`setup_vector_db.py` and any ingestion scripts).
    - Consider adding a migration note to the README if old vector DBs will become incompatible.

- **Context size and performance**
  - Any new context-building logic should:
    - Deduplicate text aggressively.
    - Prefer higher-level summaries over many small overlapping chunks.
    - Avoid unnecessary calls to Gemini inside tight loops.

- **Testing and safety**
  - Prefer to:
    - Run small, representative test questions via `chunk_tree_retriever.py "Some question"` after changes to retrieval or context building.
    - Avoid destructive operations on `data/` and `cache/` unless explicitly requested (use `ingest_pdfs.py --clear-cache` if a clean slate is needed).

- **Style**
  - Follow existing Python style (type hints, small focused functions).
  - Avoid adding verbose comments that simply restate what the code does; comment only on non-obvious intent and trade-offs.

