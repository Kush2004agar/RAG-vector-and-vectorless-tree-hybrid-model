import chromadb
import json
from pathlib import Path
from config import VECTOR_DB_DIR, TREES_DIR


def _valid_text(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("error:"):
        return False
    return True


def setup_databases():
    """
    Initializes ChromaDB collections for the Tree.
    Requires one collection for filtering Root Summaries, 
    one for Parents, and one for Child chunks.
    """
    print(f"Initializing Vector DB at {VECTOR_DB_DIR}")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # 1. Root Summaries DB (for filter Phase 1)
    root_collection = client.get_or_create_collection(name="root_summaries")

    # 2. Parent Summaries DB (for Phase 2 semantic search over sections)
    parents_collection = client.get_or_create_collection(name="parent_summaries")

    # 3. Child Chunks DB (for Phase 3 actual retrieval, if semantic search needed,
    # but we can also just use an SQLite/JSON lookup DB by chunk_id)
    chunks_collection = client.get_or_create_collection(name="child_chunks")

    tree_files = list(TREES_DIR.glob("*.json"))
    if not tree_files:
        print("No trees found in cache. Build them first.")
        return root_collection, parents_collection, chunks_collection
        
    for f in tree_files:
        with open(f, 'r') as file:
            tree = json.load(file)
            
        pdf_name = tree['doc_name']

        # Add Root summary
        if _valid_text(tree.get("root_summary", "")):
            root_collection.upsert(
                documents=[tree['root_summary']],
                metadatas=[{"pdf_name": pdf_name}],
                ids=[f"root_{pdf_name}"]
            )
        else:
            print(f"Skipping invalid root summary for {pdf_name}.")

        # Add Parent summaries
        parent_docs = []
        parent_metas = []
        parent_ids = []
        chunk_to_parent = {}

        for p in tree.get("parents", []):
            if not _valid_text(p.get("summary", "")):
                continue
            parent_docs.append(p["summary"])
            parent_metas.append(
                {
                    "pdf_name": pdf_name,
                    "parent_id": p["parent_id"],
                }
            )
            parent_ids.append(f"parent_{pdf_name}_{p['parent_id']}")
            for chunk_id in p.get("child_chunk_ids", []):
                chunk_to_parent[str(chunk_id)] = p["parent_id"]

        if parent_docs:
            parents_collection.upsert(
                documents=parent_docs,
                metadatas=parent_metas,
                ids=parent_ids,
            )

        # Add actual child chunks
        chunk_docs = []
        chunk_metas = []
        chunk_ids = []

        for chunk in tree.get("chunks", []):
            if not _valid_text(chunk.get("text", "")):
                continue
            chunk_docs.append(chunk["text"])
            chunk_metas.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "file_name": chunk["file_name"],
                    "page": chunk.get("page", 0),
                    "pdf_name": pdf_name,
                    "parent_id": chunk_to_parent.get(str(chunk["chunk_id"]), ""),
                }
            )
            chunk_ids.append(str(chunk["chunk_id"]))

        if chunk_docs:
            chunks_collection.upsert(
                documents=chunk_docs,
                metadatas=chunk_metas,
                ids=chunk_ids,
            )

    print(f"Vector DB populated with {len(tree_files)} trees.")
    return root_collection, parents_collection, chunks_collection

if __name__ == "__main__":
    setup_databases()
