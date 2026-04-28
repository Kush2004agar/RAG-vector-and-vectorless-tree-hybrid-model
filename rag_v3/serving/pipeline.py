from __future__ import annotations

import os

from google import genai

from rag_v3.context.context_builder import ContextBuilder
from rag_v3.ingestion.feature_extractor import FeatureExtractor
from rag_v3.indexing.embedder import Embedder
from rag_v3.indexing.qdrant_client import QdrantStore
from rag_v3.ranking.reranker import Reranker
from rag_v3.retrieval.filter_builder import FilterBuilder
from rag_v3.retrieval.query_processor import QueryProcessor
from rag_v3.retrieval.retriever import Retriever

NOT_FOUND_MESSAGE = "Information not found in available documents."


class RagV3Pipeline:
    """Single pipeline: process -> filter -> Qdrant search -> rerank -> context -> LLM."""

    def __init__(self) -> None:
        feature_extractor = FeatureExtractor()
        embedder = Embedder()
        store = QdrantStore(vector_size=embedder.dim)

        self.query_processor = QueryProcessor(feature_extractor)
        self.retriever = Retriever(store=store, embedder=embedder, filter_builder=FilterBuilder())
        self.reranker = Reranker()
        self.context_builder = ContextBuilder(min_chunks=5, max_chunks=8)

        api_key = os.environ.get("GEMINI_API_KEY")
        self.llm = genai.Client(api_key=api_key) if api_key else None

    def generate_answer(self, question: str, context: str) -> str:
        if not context:
            return NOT_FOUND_MESSAGE

        if not self.llm:
            return f"{NOT_FOUND_MESSAGE} (LLM unavailable: GEMINI_API_KEY not configured)"

        prompt = (
            "Answer the question using only the provided context.\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            f"If context is insufficient, reply exactly: {NOT_FOUND_MESSAGE}"
        )
        response = self.llm.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.0),
        )
        return (response.text or "").strip() or NOT_FOUND_MESSAGE

    def answer(self, question: str) -> str:
        query = self.query_processor.process(question)
        candidates = self.retriever.retrieve(query, limit=50)
        reranked = self.reranker.rerank(query, candidates)
        context = self.context_builder.build(reranked)
        return self.generate_answer(question=query.cleaned, context=context)
