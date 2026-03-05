import os
import json
from google import genai
from typing import List, Dict
from pathlib import Path
from config import GEMINI_API_KEY, CHUNK_GROUP_SIZE, CACHE_DIR, TREES_DIR

# Setup Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_summary(text_list: List[str], level: str) -> str:
    """Generates a concise summary for a group of chunks or a document."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key":
         print("Warning: GEMINI_API_KEY not set. Generating mock summary.")
         return f"[MOCK {level.upper()} SUMMARY] Focus topics: AI, Network Security, 5G"
         
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
        return f"Error: Could not generate summary for {level}"

def build_chunk_tree(pdf_name: str, all_chunks: List[Dict]):
    """
    Groups existing chunks into 'Parent Nodes' and creates a tree.
    """
    print(f"Building Chunk-Tree for {pdf_name} ({len(all_chunks)} chunks)...")
    tree = {
        "doc_name": pdf_name,
        "root_summary": "",
        "parents": [],
        "chunks": all_chunks # In phase 3 we pull exact child chunks, we can store them here or refer to DB
    }
    
    parent_summaries = []
    
    for i in range(0, len(all_chunks), CHUNK_GROUP_SIZE):
        group = all_chunks[i : i + CHUNK_GROUP_SIZE]
        group_texts = [c['text'] for c in group]
        
        parent_summary = generate_summary(group_texts, "section/group of chunks")
        parent_summaries.append(parent_summary)
        
        tree["parents"].append({
            "parent_id": f"{pdf_name}_p_{i//CHUNK_GROUP_SIZE}",
            "summary": parent_summary,
            "child_chunk_ids": [c['chunk_id'] for c in group]
        })
        
    # Generate the Root Summary for the whole PDF
    if parent_summaries:
        print("Generating Root Summary...")
        tree["root_summary"] = generate_summary(parent_summaries, "entire document")
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
