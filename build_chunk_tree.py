import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from typing import List, Dict, Optional, Tuple
from config import GEMINI_API_KEY, CHUNK_GROUP_SIZE, CACHE_DIR, TREES_DIR

# Setup Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Model name constant (easy to update in one place)
_GEMINI_MODEL = "gemini-2.5-flash"

# Pre-check API key validity once at module load
_API_KEY_VALID = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key")

# Max workers for concurrent Gemini API calls
_MAX_WORKERS = 5

# Retry settings for transient API failures
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


def _is_bad_summary(text: str) -> bool:
    t = (text or "").strip().lower()
    return not t or t.startswith("error:") or t.startswith("[mock")


def _fallback_summary(text_list: List[str], max_chars: int = 420) -> str:
    combined = " ".join((t or "").strip() for t in text_list if (t or "").strip())
    if not combined:
        return "No meaningful content available."
    return combined[:max_chars].strip()


def generate_summary(text_list: List[str], level: str) -> str:
    """Generates a concise summary for a group of chunks or a document."""
    if not _API_KEY_VALID:
        print("Warning: GEMINI_API_KEY not set. Using fallback summary.")
        return _fallback_summary(text_list)

    combined_text = "\n\n".join(text_list)[:4000]  # Limit context
    prompt = (
        f"Summarize the following {level} of a document. "
        f"Focus on the core topics and technical keywords: \n\n {combined_text}"
    )

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.0),
            )
            return response.text.strip()
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                print(f"Retrying summary generation (attempt {attempt + 1}): {e}")
                time.sleep(delay)

    print(f"Error generating summary after {_MAX_RETRIES} attempts: {last_exc}")
    return _fallback_summary(text_list)


def _summarize_group(group: List[Dict], level_label: str, node_id: str, id_key: str = "node_id") -> Dict:
    """Summarizes a single group of nodes and returns a parent node dict."""
    texts = [n["summary"] for n in group]
    summary = generate_summary(texts, level_label)
    if _is_bad_summary(summary):
        summary = _fallback_summary(texts)
    return {
        "node_id": node_id,
        "summary": summary,
        "children": [n[id_key] for n in group],
    }


def _build_recursive_levels(parent_nodes: List[Dict], pdf_name: str) -> List[List[Dict]]:
    """
    Recursively groups summaries to create a hierarchical tree above parent nodes.
    Groups within each level are summarised concurrently.
    """
    levels: List[List[Dict]] = []
    current_nodes = [
        {"node_id": node["parent_id"], "summary": node["summary"], "child_ids": node["child_chunk_ids"]}
        for node in parent_nodes
    ]
    level_idx = 1

    while len(current_nodes) > 1:
        groups = [
            (current_nodes[i : i + CHUNK_GROUP_SIZE], i)
            for i in range(0, len(current_nodes), CHUNK_GROUP_SIZE)
        ]

        next_level: List[Optional[Dict]] = [None] * len(groups)

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(groups))) as executor:
            futures = {
                executor.submit(
                    _summarize_group,
                    group,
                    f"hierarchy level {level_idx}",
                    f"{pdf_name}_h_{level_idx}_{i // CHUNK_GROUP_SIZE}",
                ): idx
                for idx, (group, i) in enumerate(groups)
            }
            for future in as_completed(futures):
                idx = futures[future]
                next_level[idx] = future.result()

        levels.append([n for n in next_level if n is not None])
        current_nodes = [n for n in next_level if n is not None]
        level_idx += 1

    return levels


def build_chunk_tree(pdf_name: str, all_chunks: List[Dict]) -> Dict:
    """
    Groups existing chunks into 'Parent Nodes' and creates a tree.
    Parent summaries are generated concurrently.
    """
    print(f"Building Chunk-Tree for {pdf_name} ({len(all_chunks)} chunks)...")
    tree = {
        "doc_name": pdf_name,
        "root_summary": "",
        "parents": [],
        "hierarchy_levels": [],
        "chunks": all_chunks,  # In phase 3 we pull exact child chunks, we can store them here or refer to DB
    }

    # Prepare groups
    groups = [
        (all_chunks[i : i + CHUNK_GROUP_SIZE], i)
        for i in range(0, len(all_chunks), CHUNK_GROUP_SIZE)
    ]

    if not groups:
        tree["root_summary"] = "No content extracted."
        return tree

    # Summarize parent groups concurrently
    parent_results: List[Optional[Dict]] = [None] * len(groups)

    def _summarize_parent(group: List[Dict], group_idx: int) -> Tuple[int, str, List[str]]:
        group_texts = [c["text"] for c in group]
        summary = generate_summary(group_texts, "section/group of chunks")
        if _is_bad_summary(summary):
            summary = _fallback_summary(group_texts)
        chunk_ids = [c["chunk_id"] for c in group]
        return group_idx, summary, chunk_ids

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(groups))) as executor:
        futures = {
            executor.submit(_summarize_parent, group, idx): (idx, i)
            for idx, (group, i) in enumerate(groups)
        }
        for future in as_completed(futures):
            idx, i = futures[future]
            group_idx, summary, chunk_ids = future.result()
            parent_results[group_idx] = {
                "parent_id": f"{pdf_name}_p_{i // CHUNK_GROUP_SIZE}",
                "summary": summary,
                "child_chunk_ids": chunk_ids,
            }

    tree["parents"] = [p for p in parent_results if p is not None]
    parent_summaries = [p["summary"] for p in tree["parents"]]

    # Build hierarchy levels and root summary
    tree["hierarchy_levels"] = _build_recursive_levels(tree["parents"], pdf_name)
    print("Generating Root Summary...")
    if tree["hierarchy_levels"]:
        top_summaries = [n["summary"] for n in tree["hierarchy_levels"][-1]]
        tree["root_summary"] = generate_summary(top_summaries, "entire document")
    else:
        tree["root_summary"] = generate_summary(parent_summaries, "entire document")
    if _is_bad_summary(tree["root_summary"]):
        tree["root_summary"] = _fallback_summary(parent_summaries)

    return tree


def _build_and_save_tree(pdf_name: str, chunks: List[Dict]) -> None:
    """Builds a chunk tree for one document and persists it to disk."""
    tree_file = TREES_DIR / f"{pdf_name}.json"
    if tree_file.exists():
        print(f"Tree already exists for {pdf_name}. Skipping.")
        return
    tree = build_chunk_tree(pdf_name, chunks)
    with open(tree_file, "w") as f:
        json.dump(tree, f, indent=2)


def process_all_trees() -> None:
    """Reads raw chunks and builds trees for any document without one.
    Documents are processed concurrently.
    """
    raw_chunks_file = CACHE_DIR / "raw_chunks.json"
    if not raw_chunks_file.exists():
        print("No raw chunks found. Run ingest_pdfs.py first.")
        return

    with open(raw_chunks_file, "r") as f:
        all_chunks = json.load(f)

    pending = {
        pdf_name: chunks
        for pdf_name, chunks in all_chunks.items()
        if not (TREES_DIR / f"{pdf_name}.json").exists()
    }

    already_done = len(all_chunks) - len(pending)
    if already_done:
        print(f"Skipping {already_done} already-built trees.")

    if not pending:
        print("All trees are up to date.")
        return

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(pending))) as executor:
        futures = {
            executor.submit(_build_and_save_tree, pdf_name, chunks): pdf_name
            for pdf_name, chunks in pending.items()
        }
        for future in as_completed(futures):
            pdf_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error building tree for {pdf_name}: {e}")

    print(f"Finished building {len(pending)} chunk trees.")


if __name__ == "__main__":
    process_all_trees()
