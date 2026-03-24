# Code Review Notes (2026-03-24)

## High-priority findings

1. **`--clear-cache` does not clear `cache/raw_chunks.json`**, so stale chunks can be merged back in on the same run.
   - `clear_cache()` removes vector DB, trackers, and `cache/chunk_trees`, but does not remove `raw_chunks.json`.
   - The main flow then reloads `raw_chunks.json` and merges `existing_chunks.update(all_chunks)`, which can reintroduce old document entries.

2. **`generate_answer()` has no missing-key guard in retriever path**, unlike summarization path.
   - `chunk_tree_retriever.py` always initializes Gemini client and directly calls the API.
   - If `GEMINI_API_KEY` is unset/placeholder, answer generation fails at runtime instead of using a deterministic fallback.

## Medium-priority findings

1. **Duplicate log line in `answer_question()`**
   - `Phase 3: Context filtering and compression...` is printed twice.

2. **Several unused imports add noise**
   - `ingest_pdfs.py` imports `os`, `glob`, and `Path` where some are unused.
   - `build_chunk_tree.py` imports `os` and `Path` but does not use them.
   - `setup_vector_db.py` imports `Path` but does not use it.

## Suggested next changes

1. In `clear_cache()`, remove `cache/raw_chunks.json` when `--clear-cache` is passed.
2. Add a no-key fallback path to `chunk_tree_retriever.generate_answer()` similar to `build_chunk_tree.generate_summary()`.
3. Remove duplicate Phase 3 print and clean unused imports.
