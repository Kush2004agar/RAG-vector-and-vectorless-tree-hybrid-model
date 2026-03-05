import chromadb
import json
from google import genai
from pathlib import Path
from typing import List, Dict

from config import VECTOR_DB_DIR, TREES_DIR, GEMINI_API_KEY, RETRIEVER_TOP_K

# Setup Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Setup Chroma client
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
try:
    root_collection = chroma_client.get_collection(name="root_summaries")
    parents_collection = chroma_client.get_collection(name="parent_summaries")
    chunks_collection = chroma_client.get_collection(name="child_chunks")
except Exception as e:
    print("Warning: Vector DB not fully initialized. Run setup_vector_db.py first.")
    root_collection = None
    parents_collection = None
    chunks_collection = None

def load_tree(pdf_name: str) -> Dict:
    """Loads a specific document's tree from disk."""
    tree_path = TREES_DIR / f"{pdf_name}.json"
    if tree_path.exists():
        with open(tree_path, 'r') as f:
            return json.load(f)
    return {}

def answer_question(question: str) -> str:
    """Runs the 3-Phase reasoning layer to answer a question."""
    if not root_collection:
        return "System error: Database not initialized."
        
    print(f"\n--- Processing Question: {question} ---")
    
    # --- PHASE 1 (The Filter) ---
    print("Phase 1: Finding relevant Root Summaries...")
    root_results = root_collection.query(
        query_texts=[question],
        n_results=RETRIEVER_TOP_K  # Use configurable top-k
    )
    
    top_pdf_names = [meta["pdf_name"] for meta in root_results["metadatas"][0]]
    if not top_pdf_names:
         return "No relevant documents found."
         
    print(f"Top 3 PDFs selected: {top_pdf_names}")
    
    # --- PHASE 2 (The Reasoner) ---
    print("Phase 2: Reasoning over Parent Summaries...")
    selected_parent_ids = set()

    if parents_collection:
        # Vector-search parents per selected PDF instead of sending all to the LLM
        for pdf_name in top_pdf_names:
            try:
                parent_results = parents_collection.query(
                    query_texts=[question],
                    n_results=RETRIEVER_TOP_K,
                    where={"pdf_name": pdf_name},
                )
                for meta in parent_results.get("metadatas", [[]])[0]:
                    pid = meta.get("parent_id")
                    if pid:
                        selected_parent_ids.add(pid)
            except Exception as e:
                print(f"Error querying parent_summaries for {pdf_name}: {e}")
    else:
        # Fallback to original LLM-based reasoning over all parents
        for pdf_name in top_pdf_names:
            tree = load_tree(pdf_name)
            if not tree:
                continue

            parent_summaries = tree.get("parents", [])

            parents_text = ""
            for p in parent_summaries:
                parents_text += f"Parent ID: {p['parent_id']}\nSummary: {p['summary']}\n\n"

            prompt = (
                f"Question: {question}\n\n"
                f"Below are section summaries for the document '{pdf_name}'. "
                "Based ONLY on the question and these summaries, which Parent IDs "
                "are most likely to contain the specific technical answer? "
                "Return ONLY a comma-separated list of Parent IDs. If none seem relevant, return 'NONE'.\n\n"
                f"Summaries:\n{parents_text}"
            )

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(temperature=0.0),
                )
                llm_reply = response.text.strip()
                if "NONE" not in llm_reply.upper():
                    ids = [x.strip() for x in llm_reply.split(",") if x.strip()]
                    for pid in ids:
                        selected_parent_ids.add(pid)
            except Exception as e:
                print(f"Error reasoning over {pdf_name}: {e}")

    selected_parent_ids = list(selected_parent_ids)
    print(f"Selected Parent IDs: {selected_parent_ids}")
    if not selected_parent_ids:
        return "Could not identify specific sections containing the answer."
        
    # --- PHASE 3 (The Extractor) ---
    print("Phase 3: Extracting specific Child Chunks...")
    final_context_chunks = []
    
    # We have parent ids, now resolve them to absolute child chunk ids.
    for pdf_name in top_pdf_names:
        tree = load_tree(pdf_name)
        for p in tree.get("parents", []):
            if p["parent_id"] in selected_parent_ids:
                # We fetch exact chunks by chunk_id from Chroma or our JSON dictionary
                child_ids = p["child_chunk_ids"]
                child_ids = [str(cid) for cid in child_ids]
                
                # We pull these exact chunks from the chunks_collection
                if child_ids:
                     chunk_data = chunks_collection.get(ids=child_ids)
                     for i, doc in enumerate(chunk_data["documents"]):
                         final_context_chunks.append(f"[{pdf_name}] {doc}")
                         
    print(f"Extracted {len(final_context_chunks)} specific chunks. Generating final answer...")
    
    # --- PHASE 4: Final Answer Generation ---
    final_context_text = "\n\n".join(final_context_chunks)
    
    final_prompt = (
         f"Answer the following question using ONLY the provided context.\n"
         f"Question: {question}\n\n"
         f"Context:\n{final_context_text}\n\n"
         "Provide a highly specific, technical answer directly addressing the question. "
         "If the answer is not contained in the context, say 'Information not found in available documents.'"
    )
    
    try:
        final_response = client.models.generate_content(
            model='gemini-2.5-pro', # Use a more powerful model for final reasoning accuracy if available, or stay on flash
            contents=final_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0
            )
        )
        return final_response.text.strip()
    except Exception as e:
        return f"Error generating final answer: {e}"

if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "What is the difference between SSH and TLS?"
    ans = answer_question(test_q)
    print("\nFINAL ANSWER:\n")
    print(ans)
