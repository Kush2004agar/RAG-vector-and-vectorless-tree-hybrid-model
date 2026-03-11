import chromadb
import json
import re
from collections import OrderedDict
from google import genai
from threading import Lock
from typing import Dict, List

from config import (
    VECTOR_DB_DIR,
    TREES_DIR,
    GEMINI_API_KEY,
    ROOT_RETRIEVER_TOP_K,
    PARENT_RETRIEVER_TOP_K,
    DIRECT_CHUNK_RETRIEVER_TOP_K,
    INITIAL_CONTEXT_CHUNK_COUNT,
    FALLBACK_CONTEXT_CHUNK_COUNT,
    MAX_PARENT_CANDIDATES,
    ENABLE_QUERY_ROUTER,
    ENABLE_RERANKING,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_POOL_SIZE,
    RERANKED_TOP_K,
    CONTEXT_COMPRESSION_MAX_CHARS,
    CONTEXT_MIN_RELEVANCE_SCORE,
    ENABLE_RETRIEVAL_CACHE,
    RETRIEVAL_CACHE_SIZE,
)

NOT_FOUND_MESSAGE = "Information not found in available documents."
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}

# Setup Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Setup Chroma client
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
try:
    root_collection = chroma_client.get_collection(name="root_summaries")
    parents_collection = chroma_client.get_collection(name="parent_summaries")
    chunks_collection = chroma_client.get_collection(name="child_chunks")
except Exception:
    print("Warning: Vector DB not fully initialized. Run setup_vector_db.py first.")
    root_collection = None
    parents_collection = None
    chunks_collection = None

RETRIEVAL_CACHE: OrderedDict[str, Dict] = OrderedDict()
RETRIEVAL_CACHE_LOCK = Lock()


def _clone_jsonable(value):
    return json.loads(json.dumps(value))


def _cache_key(question: str) -> str:
    return " ".join((question or "").strip().lower().split())


def _cache_get(question: str) -> Dict | None:
    if not ENABLE_RETRIEVAL_CACHE:
        return None
    key = _cache_key(question)
    with RETRIEVAL_CACHE_LOCK:
        if key not in RETRIEVAL_CACHE:
            return None
        RETRIEVAL_CACHE.move_to_end(key)
        return _clone_jsonable(RETRIEVAL_CACHE[key])


def _cache_set(question: str, payload: Dict):
    if not ENABLE_RETRIEVAL_CACHE:
        return
    key = _cache_key(question)
    with RETRIEVAL_CACHE_LOCK:
        RETRIEVAL_CACHE[key] = _clone_jsonable(payload)
        RETRIEVAL_CACHE.move_to_end(key)
        while len(RETRIEVAL_CACHE) > RETRIEVAL_CACHE_SIZE:
            RETRIEVAL_CACHE.popitem(last=False)


def _route_query(question: str) -> str:
    """
    Lightweight router for retrieval strategy selection.
    Returns one of: 'tree', 'vector', 'lexical', 'hybrid'.
    """
    if not ENABLE_QUERY_ROUTER:
        return "hybrid"

    q = (question or "").lower()
    tokens = tokenize(question)

    lexical_markers = ("exact", "verbatim", "quote", "page", "section", "clause", "table")
    tree_markers = ("overview", "high-level", "summary", "summarize", "architecture")
    if any(marker in q for marker in lexical_markers):
        return "lexical"
    if any(marker in q for marker in tree_markers):
        return "tree"
    if len(tokens) <= 4:
        return "tree"
    if len(tokens) >= 12:
        return "vector"
    return "hybrid"


def _compress_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    left = max_chars // 2
    right = max_chars - left - 5
    return f"{text[:left].rstrip()} ... {text[-right:].lstrip()}"


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.load_error = ""
        self._load_attempted = False

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        if not ENABLE_RERANKING:
            self.load_error = "Reranking disabled by config."
            return

        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            self.load_error = str(e)

    def score(self, question: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        self._ensure_loaded()
        if self.model is None:
            return candidates

        pairs = [(question, c.get("text", "")[:2000]) for c in candidates]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            self.load_error = str(e)
            return candidates

        rescored = []
        for c, score in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(score)
            # Keep lexical/vector score and add reranker influence.
            item["score"] = item.get("score", 0.0) + (float(score) * 2.0)
            rescored.append(item)
        return rescored


reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)


def load_tree(pdf_name: str) -> Dict:
    """Load a specific document tree from disk."""
    tree_path = TREES_DIR / f"{pdf_name}.json"
    if tree_path.exists():
        with open(tree_path, "r") as f:
            return json.load(f)
    return {}


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_+-]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def lexical_score(question: str, text: str) -> float:
    question_tokens = tokenize(question)
    text_tokens = set(tokenize(text))
    if not question_tokens:
        return 0.0

    overlap = sum(1 for token in question_tokens if token in text_tokens)
    coverage = overlap / len(question_tokens)

    score = overlap * 2.0 + coverage * 5.0

    lowered_text = text.lower()
    if "ssh" in question.lower() and "ssh" in lowered_text:
        score += 2.0
    if "tls" in question.lower() and "tls" in lowered_text:
        score += 2.0
    if "ipsec" in question.lower() and "ipsec" in lowered_text:
        score += 2.0

    return score


def combined_score(question: str, text: str, distance=None) -> float:
    score = lexical_score(question, text)
    if distance is not None:
        score += max(0.0, 3.0 - float(distance))
    return score


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def query_root_documents(question: str) -> List[str]:
    root_results = root_collection.query(
        query_texts=[question],
        n_results=ROOT_RETRIEVER_TOP_K,
    )
    metadatas = root_results.get("metadatas", [[]])[0]
    return dedupe_preserve_order([meta.get("pdf_name") for meta in metadatas if meta])


def get_parent_candidates(question: str, pdf_name: str) -> List[Dict]:
    candidates = []

    if parents_collection:
        try:
            results = parents_collection.query(
                query_texts=[question],
                n_results=PARENT_RETRIEVER_TOP_K,
                where={"pdf_name": pdf_name},
            )
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for meta, document, distance in zip(metadatas, documents, distances):
                parent_id = meta.get("parent_id")
                if parent_id:
                    candidates.append(
                        {
                            "parent_id": parent_id,
                            "score": combined_score(question, document, distance),
                        }
                    )
        except Exception as e:
            print(f"Error querying parent_summaries for {pdf_name}: {e}")

    if candidates:
        return candidates

    tree = load_tree(pdf_name)
    for parent in tree.get("parents", []):
        candidates.append(
            {
                "parent_id": parent["parent_id"],
                "score": lexical_score(question, parent.get("summary", "")),
            }
        )
    return candidates


def add_candidate(candidate_map: Dict[str, Dict], candidate: Dict):
    key = f"{candidate.get('pdf_name', '')}::{candidate.get('chunk_id', '')}"
    candidate["candidate_key"] = key
    existing = candidate_map.get(key)
    if not existing or candidate["score"] > existing["score"]:
        candidate_map[key] = candidate


def _chunk_text_from_tree(tree: Dict, chunk_id: str) -> tuple:
    """Get (text, page) for chunk_id from tree chunks. Returns ('', 0) if not found."""
    for c in tree.get("chunks", []):
        if str(c.get("chunk_id", "")) == str(chunk_id):
            return (c.get("text") or "", c.get("page", 0))
    return ("", 0)


def get_chunks_from_selected_parents(question: str, top_pdf_names: List[str], selected_parent_ids: List[str]) -> Dict[str, Dict]:
    candidate_map = {}
    selected_parent_id_set = set(selected_parent_ids)

    for pdf_name in top_pdf_names:
        tree = load_tree(pdf_name)
        if not tree:
            continue

        for parent in tree.get("parents", []):
            if parent["parent_id"] not in selected_parent_id_set:
                continue

            child_ids = [str(cid) for cid in parent.get("child_chunk_ids", [])]
            if not child_ids:
                continue

            for cid in child_ids:
                text, page = _chunk_text_from_tree(tree, cid)
                if not (text and text.strip()):
                    continue
                add_candidate(
                    candidate_map,
                    {
                        "chunk_id": cid,
                        "pdf_name": pdf_name,
                        "page": page,
                        "text": text,
                        "score": lexical_score(question, text) + 1.5,
                        "source": "tree",
                    },
                )

    return candidate_map


def get_direct_chunk_candidates(question: str, top_pdf_names: List[str]) -> Dict[str, Dict]:
    candidate_map = {}

    if not chunks_collection:
        return candidate_map

    for pdf_name in top_pdf_names:
        try:
            results = chunks_collection.query(
                query_texts=[question],
                n_results=DIRECT_CHUNK_RETRIEVER_TOP_K,
                where={"pdf_name": pdf_name},
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for metadata, document, distance in zip(metadatas, documents, distances):
                if document is None or not (str(document or "").strip()):
                    continue
                chunk_id = str(metadata.get("chunk_id", ""))
                add_candidate(
                    candidate_map,
                    {
                        "chunk_id": chunk_id,
                        "pdf_name": metadata.get("pdf_name", pdf_name),
                        "page": metadata.get("page", 0),
                        "text": document,
                        "score": combined_score(question, document, distance),
                        "source": "vector",
                    },
                )
        except Exception as e:
            print(f"Error querying child_chunks for {pdf_name}: {e}")

    return candidate_map


def get_global_chunk_candidates(question: str) -> Dict[str, Dict]:
    candidate_map = {}

    if not chunks_collection:
        return candidate_map

    try:
        results = chunks_collection.query(
            query_texts=[question],
            n_results=FALLBACK_CONTEXT_CHUNK_COUNT,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for metadata, document, distance in zip(metadatas, documents, distances):
            if document is None or not (str(document or "").strip()):
                continue
            chunk_id = str(metadata.get("chunk_id", ""))
            add_candidate(
                candidate_map,
                {
                    "chunk_id": chunk_id,
                    "pdf_name": metadata.get("pdf_name", ""),
                    "page": metadata.get("page", 0),
                    "text": document,
                    "score": combined_score(question, document, distance),
                    "source": "vector_global",
                },
            )
    except Exception as e:
        print(f"Error running fallback chunk search: {e}")

    return candidate_map


def rank_candidates(candidate_map_or_list, limit: int) -> List[Dict]:
    if isinstance(candidate_map_or_list, dict):
        values = candidate_map_or_list.values()
    else:
        values = candidate_map_or_list

    ranked = sorted(
        values,
        key=lambda item: (
            item.get("rerank_score", 0.0),
            item.get("score", 0.0),
            len(item.get("text", "")),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _apply_route_bias(question: str, route: str, candidates: Dict[str, Dict]) -> Dict[str, Dict]:
    if route == "hybrid":
        return candidates

    for c in candidates.values():
        source = c.get("source", "")
        if route == "tree":
            if source == "tree":
                c["score"] += 1.5
        elif route == "vector":
            if source in {"vector", "vector_global"}:
                c["score"] += 1.2
        elif route == "lexical":
            c["score"] += lexical_score(question, c.get("text", "")) * 0.2
            if source == "tree":
                c["score"] -= 0.25
    return candidates


def _rerank_candidates(question: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    if not candidates:
        return []
    rescored = reranker.score(question, candidates)
    return rank_candidates(rescored, top_k)


def _filter_and_compress_context(question: str, chunks: List[Dict], limit: int) -> List[Dict]:
    filtered = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text or not str(text).strip():
            continue
        relevance = lexical_score(question, text)
        if relevance >= CONTEXT_MIN_RELEVANCE_SCORE or chunk.get("rerank_score") is not None:
            item = dict(chunk)
            item["text"] = _compress_text(text, CONTEXT_COMPRESSION_MAX_CHARS)
            item["relevance_score"] = relevance
            filtered.append(item)

    if not filtered:
        # If threshold was too strict, keep top chunks anyway.
        fallback = rank_candidates(chunks, limit)
        for chunk in fallback:
            item = dict(chunk)
            item["text"] = _compress_text(item.get("text", ""), CONTEXT_COMPRESSION_MAX_CHARS)
            item["relevance_score"] = lexical_score(question, item.get("text", ""))
            filtered.append(item)

    return filtered[:limit]


def build_context(question: str, top_pdf_names: List[str], route: str = "hybrid") -> Dict:
    all_parent_candidates = []
    for pdf_name in top_pdf_names:
        all_parent_candidates.extend(get_parent_candidates(question, pdf_name))

    all_parent_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected_parent_ids = dedupe_preserve_order(
        [item["parent_id"] for item in all_parent_candidates]
    )[:MAX_PARENT_CANDIDATES]

    parent_chunk_candidates = get_chunks_from_selected_parents(question, top_pdf_names, selected_parent_ids)
    direct_chunk_candidates = get_direct_chunk_candidates(question, top_pdf_names)
    global_chunk_candidates = get_global_chunk_candidates(question)

    merged_candidates = dict(parent_chunk_candidates)
    for chunk_id, candidate in direct_chunk_candidates.items():
        add_candidate(merged_candidates, candidate)
    for chunk_id, candidate in global_chunk_candidates.items():
        add_candidate(merged_candidates, candidate)

    merged_candidates = _apply_route_bias(question, route, merged_candidates)
    candidate_pool_size = (
        RERANK_CANDIDATE_POOL_SIZE if ENABLE_RERANKING else INITIAL_CONTEXT_CHUNK_COUNT
    )
    candidate_pool = rank_candidates(merged_candidates, candidate_pool_size)
    reranked_chunks = _rerank_candidates(
        question,
        candidate_pool,
        RERANKED_TOP_K if ENABLE_RERANKING else INITIAL_CONTEXT_CHUNK_COUNT,
    )
    final_chunks = _filter_and_compress_context(
        question,
        reranked_chunks,
        RERANKED_TOP_K if ENABLE_RERANKING else INITIAL_CONTEXT_CHUNK_COUNT,
    )

    return {
        "route": route,
        "selected_parent_ids": selected_parent_ids,
        "candidate_pool": candidate_pool,
        "ranked_chunks": final_chunks,
    }


def format_context(chunks: List[Dict]) -> str:
    return "\n\n".join(
        f"[{chunk['pdf_name']} | page {chunk['page']}]\n{chunk['text']}"
        for chunk in chunks
    )


def retrieve_relevant_chunks(question: str) -> Dict:
    if not root_collection:
        return {
            "route": "hybrid",
            "top_pdf_names": [],
            "selected_parent_ids": [],
            "candidate_pool": [],
            "ranked_chunks": [],
        }

    cached = _cache_get(question)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    route = _route_query(question)
    top_pdf_names = query_root_documents(question)
    if not top_pdf_names:
        payload = {
            "route": route,
            "top_pdf_names": [],
            "selected_parent_ids": [],
            "candidate_pool": [],
            "ranked_chunks": [],
            "cache_hit": False,
        }
        _cache_set(question, payload)
        return payload

    context = build_context(question, top_pdf_names, route=route)
    payload = {
        "route": route,
        "top_pdf_names": top_pdf_names,
        "selected_parent_ids": context["selected_parent_ids"],
        "candidate_pool": context["candidate_pool"],
        "ranked_chunks": context["ranked_chunks"],
        "cache_hit": False,
    }
    _cache_set(question, payload)
    return payload


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    # Use only chunks that have real text
    context_chunks = [c for c in context_chunks if c.get("text") and str(c["text"]).strip()]
    if not context_chunks:
        return NOT_FOUND_MESSAGE

    final_prompt = (
        "Answer the question using ONLY the provided document excerpts. Do not add any preamble or meta phrase.\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Give a short, direct answer. Prefer 2–4 short sentences or 3–5 concise bullet points.\n"
        "- Focus only on the key points needed to answer the question; avoid long explanations or background theory.\n"
        "- Start your response directly with the answer (e.g. 'The four primary security goals are:' or the first bullet). Do NOT start with phrases like 'Based on the provided document excerpts' or 'Based on the provided documents'.\n"
        "- Base your answer on the excerpts; you may paraphrase or combine points.\n"
        f"- Only reply with exactly '{NOT_FOUND_MESSAGE}' if the excerpts clearly contain nothing related to the question.\n\n"
        f"Context:\n{format_context(context_chunks)}"
    )

    final_response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=final_prompt,
        config=genai.types.GenerateContentConfig(temperature=0.0),
    )
    answer = final_response.text.strip()
    # Remove common preambles so the answer starts with the main content
    for preamble in (
        "Based on the provided document excerpts, ",
        "Based on the provided documents, ",
        "Based on the provided excerpts, ",
        "Based on the document excerpts, ",
    ):
        if answer.startswith(preamble):
            answer = answer[len(preamble) :].strip()
            break
    return answer


def answer_question(question: str) -> str:
    """Run routed retrieval + reranking and fallback if needed."""
    if not root_collection:
        return "System error: Database not initialized."

    print(f"\n--- Processing Question: {question} ---")

    print("Phase 1: Query routing + root retrieval...")
    retrieval = retrieve_relevant_chunks(question)
    top_pdf_names = retrieval["top_pdf_names"]
    if not top_pdf_names:
        return "No relevant documents found."
    print(f"Route selected: {retrieval['route']} (cache_hit={retrieval.get('cache_hit', False)})")
    print(f"Top PDFs selected: {top_pdf_names}")

    print("Phase 2: Parent + direct retrieval and reranking...")
    selected_parent_ids = retrieval["selected_parent_ids"]
    candidate_pool = retrieval["candidate_pool"]
    ranked_chunks = retrieval["ranked_chunks"]
    print(f"Selected parent IDs: {selected_parent_ids}")
    print(f"Candidate pool size before rerank: {len(candidate_pool)}")

    print("Phase 3: Context filtering and compression...")
    usable = [c for c in ranked_chunks if c.get("text") and str(c["text"]).strip()]
    print(f"Using {len(usable)} chunks for the first answer attempt.")

    if not usable:
        print("No usable chunks from routed search; trying broader chunk search...")

    try:
        answer = generate_answer(question, usable) if usable else NOT_FOUND_MESSAGE
    except Exception as e:
        return f"Error generating final answer: {e}"

    if answer.strip() != NOT_FOUND_MESSAGE:
        return answer

    print("Retrying with broader chunk search because the first attempt returned no answer...")
    broader_candidates = {
        chunk.get("candidate_key", f"{chunk.get('pdf_name', '')}::{chunk.get('chunk_id', '')}"): dict(chunk)
        for chunk in candidate_pool
    }
    for chunk_id, candidate in get_global_chunk_candidates(question).items():
        add_candidate(broader_candidates, candidate)

    broader_ranked = rank_candidates(broader_candidates, FALLBACK_CONTEXT_CHUNK_COUNT)
    broader_reranked = _rerank_candidates(
        question,
        broader_ranked,
        min(FALLBACK_CONTEXT_CHUNK_COUNT, max(RERANKED_TOP_K, INITIAL_CONTEXT_CHUNK_COUNT)),
    )
    broader_chunks = _filter_and_compress_context(question, broader_reranked, FALLBACK_CONTEXT_CHUNK_COUNT)

    try:
        return generate_answer(question, broader_chunks)
    except Exception as e:
        return f"Error generating fallback answer: {e}"

if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "What is the difference between SSH and TLS?"
    ans = answer_question(test_q)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:\n")
    print(ans)
    print("=" * 60)
    sys.stdout.flush()
