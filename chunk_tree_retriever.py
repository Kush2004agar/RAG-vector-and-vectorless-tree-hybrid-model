import chromadb
import json
import math
import re
from collections import OrderedDict
from google import genai
from threading import Lock
from typing import Dict, List, Tuple

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
    ENABLE_MULTI_QUERY_RETRIEVAL,
    ENABLE_HYBRID_RETRIEVAL,
    MULTI_QUERY_COUNT,
    PARENTS_PER_DOCUMENT,
    BM25_TOP_K,
    VECTOR_QUERY_CACHE_SIZE,
    TREE_CACHE_SIZE,
    TOKEN_CACHE_SIZE,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_POOL_SIZE,
    RERANKED_TOP_K,
    TREE_SCORE_WEIGHT,
    SEMANTIC_SCORE_WEIGHT,
    RERANK_SCORE_WEIGHT,
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
VECTOR_QUERY_CACHE: OrderedDict[str, Dict] = OrderedDict()
VECTOR_QUERY_CACHE_LOCK = Lock()
TREE_CACHE: OrderedDict[str, Dict] = OrderedDict()
TREE_CACHE_LOCK = Lock()
TOKEN_CACHE: OrderedDict[str, List[str]] = OrderedDict()
TOKEN_CACHE_LOCK = Lock()


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


def _json_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _cached_collection_query(collection, collection_name: str, query: str, n_results: int, where: Dict | None = None) -> Dict:
    if collection is None:
        return {}

    cache_key = f"{collection_name}|{query}|{n_results}|{_json_key(where or {})}"
    with VECTOR_QUERY_CACHE_LOCK:
        if cache_key in VECTOR_QUERY_CACHE:
            VECTOR_QUERY_CACHE.move_to_end(cache_key)
            return _clone_jsonable(VECTOR_QUERY_CACHE[cache_key])

    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)

    with VECTOR_QUERY_CACHE_LOCK:
        VECTOR_QUERY_CACHE[cache_key] = _clone_jsonable(results)
        VECTOR_QUERY_CACHE.move_to_end(cache_key)
        while len(VECTOR_QUERY_CACHE) > VECTOR_QUERY_CACHE_SIZE:
            VECTOR_QUERY_CACHE.popitem(last=False)
    return results


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
            item["rerank_raw_score"] = float(score)
            item["rerank_score"] = _normalized_rerank_score(float(score))
            item["score"] = compute_total_score(item)
            rescored.append(item)
        return rescored


reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)


def load_tree(pdf_name: str) -> Dict:
    """Load a specific document tree from disk."""
    with TREE_CACHE_LOCK:
        if pdf_name in TREE_CACHE:
            TREE_CACHE.move_to_end(pdf_name)
            return TREE_CACHE[pdf_name]

    tree_path = TREES_DIR / f"{pdf_name}.json"
    if tree_path.exists():
        with open(tree_path, "r") as f:
            tree = json.load(f)
        with TREE_CACHE_LOCK:
            TREE_CACHE[pdf_name] = tree
            TREE_CACHE.move_to_end(pdf_name)
            while len(TREE_CACHE) > TREE_CACHE_SIZE:
                TREE_CACHE.popitem(last=False)
        return tree
    return {}


def tokenize(text: str) -> List[str]:
    normalized = (text or "").strip().lower()
    if not normalized:
        return []

    with TOKEN_CACHE_LOCK:
        if normalized in TOKEN_CACHE:
            TOKEN_CACHE.move_to_end(normalized)
            return TOKEN_CACHE[normalized]

    tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9_+-]+", normalized)
        if token not in STOPWORDS and len(token) > 1
    ]
    with TOKEN_CACHE_LOCK:
        TOKEN_CACHE[normalized] = tokens
        TOKEN_CACHE.move_to_end(normalized)
        while len(TOKEN_CACHE) > TOKEN_CACHE_SIZE:
            TOKEN_CACHE.popitem(last=False)
    return tokens


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


def semantic_score(distance=None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, 3.0 - float(distance))


def _normalized_rerank_score(raw_score: float) -> float:
    # Cross-encoder raw outputs can be unbounded; map to a stable 0..5 range.
    clipped = max(-20.0, min(20.0, float(raw_score)))
    return (1.0 / (1.0 + math.exp(-clipped))) * 5.0


def compute_total_score(candidate: Dict) -> float:
    tree = float(candidate.get("tree_score", 0.0))
    semantic = float(candidate.get("semantic_score", 0.0))
    rerank = float(candidate.get("rerank_score", 0.0))
    return (
        TREE_SCORE_WEIGHT * tree
        + SEMANTIC_SCORE_WEIGHT * semantic
        + RERANK_SCORE_WEIGHT * rerank
    )


def component_scores(question: str, text: str, distance=None, source: str = "") -> Tuple[float, float]:
    tree = lexical_score(question, text)
    if source == "tree":
        tree += 1.5
    elif source == "vector_global":
        tree += 0.2
    sem = semantic_score(distance)
    return tree, sem


def combined_score(question: str, text: str, distance=None, source: str = "") -> Dict:
    tree, sem = component_scores(question, text, distance=distance, source=source)
    payload = {
        "tree_score": tree,
        "semantic_score": sem,
        "rerank_score": 0.0,
    }
    payload["score"] = compute_total_score(payload)
    return payload


def build_multi_queries(question: str, route: str) -> List[str]:
    if not ENABLE_MULTI_QUERY_RETRIEVAL:
        return [question]

    tokens = tokenize(question)
    keyword_variant = " ".join(tokens[:8]).strip()
    queries = [question]

    lowered = question.lower()
    diff_match = re.search(r"difference between ([^?]+?) and ([^?]+)", lowered)
    if diff_match:
        queries.append(f"compare {diff_match.group(1).strip()} vs {diff_match.group(2).strip()}")

    if keyword_variant and keyword_variant.lower() != lowered.strip():
        queries.append(keyword_variant)

    if route in {"tree", "hybrid"} and keyword_variant:
        queries.append(f"key concepts {keyword_variant}")

    return dedupe_preserve_order([q.strip() for q in queries if q and q.strip()])[: max(1, MULTI_QUERY_COUNT)]


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]]) -> List[float]:
    if not query_tokens or not docs_tokens:
        return [0.0] * len(docs_tokens)

    n_docs = len(docs_tokens)
    avgdl = (sum(len(toks) for toks in docs_tokens) / n_docs) if n_docs else 0.0
    if avgdl <= 0:
        return [0.0] * n_docs

    # Document frequencies
    doc_freq: Dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    idf: Dict[str, float] = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)

    k1 = 1.5
    b = 0.75
    scores = []
    for tokens in docs_tokens:
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        dl = len(tokens)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            numer = tf[term] * (k1 + 1.0)
            denom = tf[term] + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf.get(term, 0.0) * (numer / denom)
        scores.append(score)
    return scores


def apply_bm25_scores(query: str, candidate_map: Dict[str, Dict], query_weight: float = 1.0):
    if not ENABLE_HYBRID_RETRIEVAL or not candidate_map:
        return

    keys = list(candidate_map.keys())
    docs_tokens = [tokenize(candidate_map[k].get("text", "")) for k in keys]
    q_tokens = tokenize(query)
    raw_scores = _bm25_scores(q_tokens, docs_tokens)
    if not raw_scores:
        return

    ranked_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:BM25_TOP_K]
    max_score = raw_scores[ranked_idx[0]] if ranked_idx and raw_scores[ranked_idx[0]] > 0 else 0.0

    for i in ranked_idx:
        if raw_scores[i] <= 0:
            continue
        key = keys[i]
        candidate = candidate_map[key]
        bm25_norm = (raw_scores[i] / max_score) * 5.0 if max_score > 0 else 0.0
        # BM25 contributes to the lexical/tree component of final scoring.
        candidate["bm25_score"] = max(float(candidate.get("bm25_score", 0.0)), bm25_norm * query_weight)
        candidate["tree_score"] = float(candidate.get("tree_score", 0.0)) + (bm25_norm * query_weight)
        candidate["score"] = compute_total_score(candidate)


def query_root_documents(question: str) -> List[str]:
    if not root_collection:
        return []
    root_results = _cached_collection_query(
        root_collection,
        "root_summaries",
        question,
        ROOT_RETRIEVER_TOP_K,
    )
    metadatas = root_results.get("metadatas", [[]])[0]
    return dedupe_preserve_order([meta.get("pdf_name") for meta in metadatas if meta])


def query_root_documents_multi(queries: List[str]) -> List[str]:
    pdf_scores: Dict[str, float] = {}
    for q_idx, query in enumerate(queries):
        query_weight = 1.0 / (q_idx + 1)
        results = query_root_documents(query)
        for rank, pdf_name in enumerate(results):
            if not pdf_name:
                continue
            rr = 1.0 / (rank + 1)
            pdf_scores[pdf_name] = pdf_scores.get(pdf_name, 0.0) + (rr * query_weight)

    ranked = sorted(pdf_scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:ROOT_RETRIEVER_TOP_K]]


def get_parent_candidates(question: str, pdf_name: str) -> List[Dict]:
    candidates = []

    if parents_collection:
        try:
            results = _cached_collection_query(
                parents_collection,
                "parent_summaries",
                question,
                PARENT_RETRIEVER_TOP_K,
                where={"pdf_name": pdf_name},
            )
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for meta, document, distance in zip(metadatas, documents, distances):
                parent_id = meta.get("parent_id")
                if parent_id:
                    comp = combined_score(question, document, distance, source="vector")
                    candidates.append(
                        {
                            "parent_id": parent_id,
                            "pdf_name": pdf_name,
                            "score": comp["score"],
                        }
                    )
        except Exception as e:
            print(f"Error querying parent_summaries for {pdf_name}: {e}")

    if candidates:
        return candidates

    tree = load_tree(pdf_name)
    for parent in tree.get("parents", []):
        comp = combined_score(question, parent.get("summary", ""), None, source="tree")
        candidates.append(
            {
                "parent_id": parent["parent_id"],
                "pdf_name": pdf_name,
                "score": comp["score"],
            }
        )
    return candidates


def add_candidate(candidate_map: Dict[str, Dict], candidate: Dict):
    key = f"{candidate.get('pdf_name', '')}::{candidate.get('chunk_id', '')}"
    candidate["candidate_key"] = key
    existing = candidate_map.get(key)
    candidate.setdefault("tree_score", 0.0)
    candidate.setdefault("semantic_score", 0.0)
    candidate.setdefault("rerank_score", 0.0)
    candidate.setdefault("bm25_score", 0.0)
    candidate["score"] = compute_total_score(candidate)
    candidate["query_hits"] = int(candidate.get("query_hits", 1))
    if not existing:
        candidate_map[key] = candidate
        return

    existing["tree_score"] = max(float(existing.get("tree_score", 0.0)), float(candidate.get("tree_score", 0.0)))
    existing["semantic_score"] = max(
        float(existing.get("semantic_score", 0.0)),
        float(candidate.get("semantic_score", 0.0)),
    )
    existing["rerank_score"] = max(float(existing.get("rerank_score", 0.0)), float(candidate.get("rerank_score", 0.0)))
    existing["bm25_score"] = max(float(existing.get("bm25_score", 0.0)), float(candidate.get("bm25_score", 0.0)))
    existing["query_hits"] = int(existing.get("query_hits", 1)) + int(candidate.get("query_hits", 1))
    existing["source"] = ",".join(
        dedupe_preserve_order(
            [s for s in (str(existing.get("source", "")) + "," + str(candidate.get("source", ""))).split(",") if s]
        )
    )
    existing["score"] = compute_total_score(existing)
    if candidate.get("text") and len(str(candidate.get("text", ""))) > len(str(existing.get("text", ""))):
        existing["text"] = candidate["text"]


def _chunk_text_from_tree(tree: Dict, chunk_id: str) -> tuple:
    """Get (text, page) for chunk_id from tree chunks. Returns ('', 0) if not found."""
    for c in tree.get("chunks", []):
        if str(c.get("chunk_id", "")) == str(chunk_id):
            return (c.get("text") or "", c.get("page", 0))
    return ("", 0)


def get_chunks_from_selected_parents(
    question: str,
    selected_parent_ids_by_pdf: Dict[str, set],
    query_weight: float = 1.0,
) -> Dict[str, Dict]:
    candidate_map = {}
    for pdf_name, allowed_parent_ids in selected_parent_ids_by_pdf.items():
        if not allowed_parent_ids:
            continue
        tree = load_tree(pdf_name)
        if not tree:
            continue

        chunk_lookup = {
            str(c.get("chunk_id", "")): (c.get("text") or "", c.get("page", 0))
            for c in tree.get("chunks", [])
        }

        for parent in tree.get("parents", []):
            if parent["parent_id"] not in allowed_parent_ids:
                continue

            child_ids = [str(cid) for cid in parent.get("child_chunk_ids", [])]
            if not child_ids:
                continue

            for cid in child_ids:
                text, page = chunk_lookup.get(cid, ("", 0))
                if not (text and text.strip()):
                    continue
                tree_component, _ = component_scores(question, text, None, source="tree")
                add_candidate(
                    candidate_map,
                    {
                        "chunk_id": cid,
                        "pdf_name": pdf_name,
                        "page": page,
                        "text": text,
                        "tree_score": tree_component * query_weight,
                        "semantic_score": 0.0,
                        "rerank_score": 0.0,
                        "source": "tree",
                        "query_hits": 1,
                    },
                )

    return candidate_map


def get_direct_chunk_candidates(
    question: str,
    top_pdf_names: List[str],
    selected_parent_ids_by_pdf: Dict[str, set],
    query_weight: float = 1.0,
) -> Dict[str, Dict]:
    candidate_map = {}

    if not chunks_collection:
        return candidate_map

    for pdf_name in top_pdf_names:
        allowed_parent_ids = selected_parent_ids_by_pdf.get(pdf_name, set())
        if not allowed_parent_ids:
            continue
        try:
            results = _cached_collection_query(
                chunks_collection,
                "child_chunks",
                question,
                DIRECT_CHUNK_RETRIEVER_TOP_K,
                where={"pdf_name": pdf_name},
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for metadata, document, distance in zip(metadatas, documents, distances):
                if document is None or not (str(document or "").strip()):
                    continue
                parent_id = str(metadata.get("parent_id", ""))
                if parent_id not in allowed_parent_ids:
                    continue
                chunk_id = str(metadata.get("chunk_id", ""))
                tree_component, semantic_component = component_scores(
                    question,
                    document,
                    distance,
                    source="vector",
                )
                add_candidate(
                    candidate_map,
                    {
                        "chunk_id": chunk_id,
                        "pdf_name": metadata.get("pdf_name", pdf_name),
                        "page": metadata.get("page", 0),
                        "text": document,
                        "tree_score": tree_component * query_weight,
                        "semantic_score": semantic_component * query_weight,
                        "rerank_score": 0.0,
                        "source": "vector",
                        "query_hits": 1,
                    },
                )
        except Exception as e:
            print(f"Error querying child_chunks for {pdf_name}: {e}")

    return candidate_map


def get_broader_chunk_candidates(
    question: str,
    top_pdf_names: List[str],
    query_weight: float = 1.0,
) -> Dict[str, Dict]:
    candidate_map = {}

    if not chunks_collection:
        return candidate_map

    for pdf_name in top_pdf_names:
        try:
            results = _cached_collection_query(
                chunks_collection,
                "child_chunks",
                question,
                FALLBACK_CONTEXT_CHUNK_COUNT,
                where={"pdf_name": pdf_name},
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for metadata, document, distance in zip(metadatas, documents, distances):
                if document is None or not (str(document or "").strip()):
                    continue
                chunk_id = str(metadata.get("chunk_id", ""))
                tree_component, semantic_component = component_scores(
                    question,
                    document,
                    distance,
                    source="vector_global",
                )
                add_candidate(
                    candidate_map,
                    {
                        "chunk_id": chunk_id,
                        "pdf_name": metadata.get("pdf_name", pdf_name),
                        "page": metadata.get("page", 0),
                        "text": document,
                        "tree_score": tree_component * query_weight,
                        "semantic_score": semantic_component * query_weight,
                        "rerank_score": 0.0,
                        "source": "vector_global",
                        "query_hits": 1,
                    },
                )
        except Exception as e:
            print(f"Error running broader chunk search for {pdf_name}: {e}")

    return candidate_map


def rank_candidates(candidate_map_or_list, limit: int) -> List[Dict]:
    if isinstance(candidate_map_or_list, dict):
        values = candidate_map_or_list.values()
    else:
        values = candidate_map_or_list

    ranked = sorted(
        values,
        key=lambda item: (
            item.get("score", 0.0),
            item.get("rerank_score", 0.0),
            item.get("query_hits", 1),
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
        source_set = {s for s in str(source).split(",") if s}
        if route == "tree":
            if "tree" in source_set:
                c["tree_score"] = float(c.get("tree_score", 0.0)) + 1.5
        elif route == "vector":
            if source_set.intersection({"vector", "vector_global"}):
                c["semantic_score"] = float(c.get("semantic_score", 0.0)) + 1.2
        elif route == "lexical":
            c["tree_score"] = float(c.get("tree_score", 0.0)) + (lexical_score(question, c.get("text", "")) * 0.2)
            if "tree" in source_set:
                c["tree_score"] = float(c.get("tree_score", 0.0)) - 0.25
        c["score"] = compute_total_score(c)
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
        rerank_component = float(chunk.get("rerank_score", 0.0))
        if relevance >= CONTEXT_MIN_RELEVANCE_SCORE or rerank_component >= 1.5:
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


def build_context(
    question: str,
    top_pdf_names: List[str],
    route: str = "hybrid",
    query_variants: List[str] | None = None,
) -> Dict:
    query_variants = query_variants or [question]

    parent_scores_by_pdf: Dict[str, Dict[str, float]] = {pdf_name: {} for pdf_name in top_pdf_names}
    for q_idx, query in enumerate(query_variants):
        query_weight = 1.0 / (q_idx + 1)
        for pdf_name in top_pdf_names:
            for candidate in get_parent_candidates(query, pdf_name):
                pid = candidate["parent_id"]
                weighted_score = float(candidate["score"]) * query_weight
                prev = parent_scores_by_pdf[pdf_name].get(pid, 0.0)
                parent_scores_by_pdf[pdf_name][pid] = max(prev, weighted_score)

    selected_parent_ids_by_pdf: Dict[str, set] = {}
    global_parent_scores: List[Tuple[str, str, float]] = []
    for pdf_name in top_pdf_names:
        ranked_for_pdf = sorted(
            parent_scores_by_pdf[pdf_name].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:PARENTS_PER_DOCUMENT]
        selected_parent_ids_by_pdf[pdf_name] = {pid for pid, _ in ranked_for_pdf}
        global_parent_scores.extend((pdf_name, pid, score) for pid, score in ranked_for_pdf)

    if len(global_parent_scores) > MAX_PARENT_CANDIDATES:
        global_parent_scores.sort(key=lambda item: item[2], reverse=True)
        allowed_pairs = {(pdf, pid) for pdf, pid, _ in global_parent_scores[:MAX_PARENT_CANDIDATES]}
        selected_parent_ids_by_pdf = {
            pdf: {pid for pid in pids if (pdf, pid) in allowed_pairs}
            for pdf, pids in selected_parent_ids_by_pdf.items()
        }

    selected_parent_ids = [
        f"{pdf_name}:{pid}"
        for pdf_name, pid_set in selected_parent_ids_by_pdf.items()
        for pid in pid_set
    ]

    merged_candidates: Dict[str, Dict] = {}
    for q_idx, query in enumerate(query_variants):
        query_weight = 1.0 / (q_idx + 1)
        parent_chunk_candidates = get_chunks_from_selected_parents(
            query,
            selected_parent_ids_by_pdf,
            query_weight=query_weight,
        )
        direct_chunk_candidates = get_direct_chunk_candidates(
            query,
            top_pdf_names,
            selected_parent_ids_by_pdf,
            query_weight=query_weight,
        )

        for candidate in parent_chunk_candidates.values():
            add_candidate(merged_candidates, candidate)
        for candidate in direct_chunk_candidates.values():
            add_candidate(merged_candidates, candidate)

        # Hybrid retrieval: score the filtered candidate pool with BM25 keyword ranking.
        apply_bm25_scores(query, merged_candidates, query_weight=query_weight)

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
        "query_variants": query_variants,
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
            "query_variants": [question],
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
    query_variants = build_multi_queries(question, route)
    top_pdf_names = query_root_documents_multi(query_variants)
    if not top_pdf_names:
        payload = {
            "route": route,
            "query_variants": query_variants,
            "top_pdf_names": [],
            "selected_parent_ids": [],
            "candidate_pool": [],
            "ranked_chunks": [],
            "cache_hit": False,
        }
        _cache_set(question, payload)
        return payload

    context = build_context(
        question,
        top_pdf_names,
        route=route,
        query_variants=query_variants,
    )
    payload = {
        "route": route,
        "query_variants": query_variants,
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
    query_variants = retrieval.get("query_variants", [question])
    if not top_pdf_names:
        return "No relevant documents found."
    print(f"Route selected: {retrieval['route']} (cache_hit={retrieval.get('cache_hit', False)})")
    print(f"Multi-query variants ({len(query_variants)}): {query_variants}")
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
    for q_idx, query in enumerate(query_variants):
        query_weight = 1.0 / (q_idx + 1)
        for candidate in get_broader_chunk_candidates(
            query,
            top_pdf_names,
            query_weight=query_weight * 0.8,
        ).values():
            add_candidate(broader_candidates, candidate)
        apply_bm25_scores(query, broader_candidates, query_weight=query_weight)

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
