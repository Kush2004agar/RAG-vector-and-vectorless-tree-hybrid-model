import json

from config import RAW_CHUNKS_FILE
from rag_v3.indexing.embedder import Embedder
from rag_v3.indexing.index_builder import IndexBuilder
from rag_v3.indexing.qdrant_client import QdrantStore
from rag_v3.ingestion.chunker import Chunk
from rag_v3.ingestion.feature_extractor import FeatureExtractor


def load_chunks() -> list[Chunk]:
    if not RAW_CHUNKS_FILE.exists():
        return []

    with open(RAW_CHUNKS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks: list[Chunk] = []
    for pdf_name, items in raw.items():
        for item in items:
            chunks.append(
                Chunk(
                    id=str(item.get("chunk_id", "")),
                    text=str(item.get("text", "")),
                    pdf_name=pdf_name,
                    section=str(item.get("section", "unknown")),
                    version="v3",
                )
            )
    return chunks


def setup_databases() -> None:
    chunks = load_chunks()
    if not chunks:
        print(f"No chunks found at {RAW_CHUNKS_FILE}. Run ingest_pdfs.py first.")
        return

    embedder = Embedder()
    store = QdrantStore(vector_size=embedder.dim)
    builder = IndexBuilder(embedder=embedder, store=store, feature_extractor=FeatureExtractor())
    builder.rebuild(chunks)
    print(f"Indexed {len(chunks)} chunks in Qdrant collection 'documents'.")


if __name__ == "__main__":
    setup_databases()
