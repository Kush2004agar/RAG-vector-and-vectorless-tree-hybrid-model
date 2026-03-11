import os
import json
from google import genai
from typing import List, Dict
from pathlib import Path
from config import GEMINI_API_KEY, CHUNK_GROUP_SIZE, CACHE_DIR, TREES_DIR

# Setup Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


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
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key":
        print("Warning: GEMINI_API_KEY not set. Using fallback summary.")
        return _fallback_summary(text_list)
         
    combined_text = "\n\n".join(text_list)[:4000]  # Limit context
    prompt = f"Summarize the following {level} of a document. Focus on the core topics and technical keywords: \n\n {combined_text}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return _fallback_summary(text_list)


def _build_recursive_levels(parent_nodes: List[Dict], pdf_name: str) -> List[List[Dict]]:
    """
    Recursively groups summaries to create a hierarchical tree above parent nodes.
    """
    levels: List[List[Dict]] = []
    current_nodes = [
        {"node_id": node["parent_id"], "summary": node["summary"], "child_ids": node["child_chunk_ids"]}
        for node in parent_nodes
    ]
    level_idx = 1

    while len(current_nodes) > 1:
        next_level = []
        for i in range(0, len(current_nodes), CHUNK_GROUP_SIZE):
            group = current_nodes[i : i + CHUNK_GROUP_SIZE]
            summary = generate_summary(
                [n["summary"] for n in group],
                f"hierarchy level {level_idx}",
            )
            if _is_bad_summary(summary):
                summary = _fallback_summary([n["summary"] for n in group])
            next_level.append(
                {
                    "node_id": f"{pdf_name}_h_{level_idx}_{i // CHUNK_GROUP_SIZE}",
                    "summary": summary,
                    "children": [n["node_id"] for n in group],
                }
            )
        levels.append(next_level)
        current_nodes = next_level
        level_idx += 1

    return levels

def build_chunk_tree(pdf_name: str, all_chunks: List[Dict]):
    """
    Groups existing chunks into 'Parent Nodes' and creates a tree.
    """
    print(f"Building Chunk-Tree for {pdf_name} ({len(all_chunks)} chunks)...")
    tree = {
        "doc_name": pdf_name,
        "root_summary": "",
        "parents": [],
        "hierarchy_levels": [],
        "chunks": all_chunks # In phase 3 we pull exact child chunks, we can store them here or refer to DB
    }
    
    parent_summaries = []
    
    for i in range(0, len(all_chunks), CHUNK_GROUP_SIZE):
        group = all_chunks[i : i + CHUNK_GROUP_SIZE]
        group_texts = [c['text'] for c in group]
        
        parent_summary = generate_summary(group_texts, "section/group of chunks")
        if _is_bad_summary(parent_summary):
            parent_summary = _fallback_summary(group_texts)
        parent_summaries.append(parent_summary)
        
        tree["parents"].append({
            "parent_id": f"{pdf_name}_p_{i//CHUNK_GROUP_SIZE}",
            "summary": parent_summary,
            "child_chunk_ids": [c['chunk_id'] for c in group]
        })
        
    # Generate the Root Summary for the whole PDF
    if parent_summaries:
        tree["hierarchy_levels"] = _build_recursive_levels(tree["parents"], pdf_name)
        print("Generating Root Summary...")
        if tree["hierarchy_levels"]:
            top_summaries = [n["summary"] for n in tree["hierarchy_levels"][-1]]
            tree["root_summary"] = generate_summary(top_summaries, "entire document")
        else:
            tree["root_summary"] = generate_summary(parent_summaries, "entire document")
        if _is_bad_summary(tree["root_summary"]):
            tree["root_summary"] = _fallback_summary(parent_summaries)
    else:
        tree["root_summary"] = "No content extracted."
        
    return tree

def process_all_trees():
    """Reads raw chunks and builds trees for any document without one."""
    raw_chunks_file = CACHE_DIR / "raw_chunks.json"
    if not raw_chunks_file.exists():
        print("No raw chunks found. Run ingest_pdfs.py first.")
        return
        
    with open(raw_chunks_file, 'r') as f:
        all_chunks = json.load(f)
        
    for pdf_name, chunks in all_chunks.items():
        tree_file = TREES_DIR / f"{pdf_name}.json"
        
        if tree_file.exists():
            print(f"Tree already exists for {pdf_name}. Skipping.")
            continue
            
        tree = build_chunk_tree(pdf_name, chunks)
        
        with open(tree_file, 'w') as f:
            json.dump(tree, f, indent=2)
            
    print(f"Finished building {len(all_chunks)} chunk trees.")

if __name__ == "__main__":
    process_all_trees()
