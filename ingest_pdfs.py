import os
import glob
import json
import shutil
from pathlib import Path
from config import (
    INPUT_DIR, CACHE_DIR, VECTOR_DB_DIR, 
    PROCESSED_FILES_TRACKER, KNOWLEDGE_GRAPH_FILE, CHAT_HISTORY_FILE
)
from pdfplumber import open as pdf_open

def clear_cache():
    """Wipes the cache: vector_db, processed files, KG, history."""
    print("Clearing cache to prevent 'Ghost Cache' issues...")
    
    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)
        VECTOR_DB_DIR.mkdir()

    for f in [PROCESSED_FILES_TRACKER, KNOWLEDGE_GRAPH_FILE, CHAT_HISTORY_FILE]:
        if f.exists():
            f.unlink()
            
    # Clear chunk trees directory
    trees_dir = CACHE_DIR / "chunk_trees"
    if trees_dir.exists():
        shutil.rmtree(trees_dir)
        trees_dir.mkdir()
        
    print("Cache cleared successfully.")

def chunk_pdf(pdf_path: Path):
    """
    Extracts text from PDF and returns a list of dictionaries.
    (This simulates standard chunking).
    """
    chunks = []
    chunk_id = 0
    
    try:
        with pdf_open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                    
                # Simple chunking by paragraph/newline for demonstration.
                # In production, use langchain RecursiveCharacterTextSplitter
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                
                for p in paragraphs:
                    chunks.append({
                        "chunk_id": f"{pdf_path.stem}_{chunk_id}",
                        "file_name": pdf_path.name,
                        "page": page_num + 1,
                        "text": p
                    })
                    chunk_id += 1
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return chunks

def ingest_pdfs(force_clear=False):
    """
    Finds all PDFs in the input directory and prepares them for the chunk tree.
    """
    if force_clear:
        clear_cache()
    
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}.")
        return []
        
    processed_files = []
    if PROCESSED_FILES_TRACKER.exists():
        with open(PROCESSED_FILES_TRACKER, 'r') as f:
            processed_files = json.load(f)
            
    all_chunks = {}
    
    print(f"Ingesting {len(pdf_files)} PDFs...")
    for pdf_path in pdf_files:
        if pdf_path.name in processed_files and not force_clear:
            print(f"Skipping {pdf_path.name} (already processed).")
            continue
            
        print(f"Chunking {pdf_path.name}...")
        chunks = chunk_pdf(pdf_path)
        all_chunks[pdf_path.name] = chunks
        processed_files.append(pdf_path.name)
        
    # Save tracker
    with open(PROCESSED_FILES_TRACKER, 'w') as f:
        json.dump(processed_files, f)
        
    return all_chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true", help="Clear all previous processing data")
    args = parser.parse_args()
    
    all_chunks = ingest_pdfs(force_clear=args.clear_cache)
    
    # Save raw chunks to disk for the next step
    raw_chunks_file = CACHE_DIR / "raw_chunks.json"
    
    existing_chunks = {}
    if raw_chunks_file.exists():
         with open(raw_chunks_file, 'r') as f:
             existing_chunks = json.load(f)
             
    existing_chunks.update(all_chunks)
    
    with open(raw_chunks_file, 'w') as f:
         json.dump(existing_chunks, f)
         
    print(f"Ingestion complete. Extracted chunks for {len(all_chunks)} new documents.")
