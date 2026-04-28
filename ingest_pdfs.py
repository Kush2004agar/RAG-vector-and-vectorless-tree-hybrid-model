import json
from pathlib import Path

from pdfplumber import open as pdf_open

from config import INPUT_DIR, RAW_CHUNKS_FILE
from rag_v3.ingestion.chunker import Chunker


def extract_pdf_text(pdf_path: Path) -> str:
    pages: list[str] = []
    with pdf_open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
    return "\n\n".join(pages)


def ingest_pdfs() -> dict[str, list[dict[str, str]]]:
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}.")
        return {}

    chunker = Chunker(max_chars=1200)
    output: dict[str, list[dict[str, str]]] = {}

    for pdf_file in pdf_files:
        text = extract_pdf_text(pdf_file)
        chunks = chunker.chunk_text(text=text, pdf_name=pdf_file.name, section="full_text", version="v3")
        output[pdf_file.name] = [
            {
                "chunk_id": c.id,
                "text": c.text,
                "section": c.section,
                "version": c.version,
            }
            for c in chunks
        ]
        print(f"Ingested {pdf_file.name}: {len(chunks)} chunks")

    return output


if __name__ == "__main__":
    chunk_map = ingest_pdfs()
    with open(RAW_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, indent=2)
    print(f"Saved chunks to {RAW_CHUNKS_FILE}")
