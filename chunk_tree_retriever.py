"""Compatibility wrapper over the V3 single-pipeline retriever."""

from rag_v3.serving.pipeline import RagV3Pipeline

_pipeline = RagV3Pipeline()


def answer_question(question: str) -> str:
    return _pipeline.answer(question)


if __name__ == "__main__":
    import sys

    test_q = sys.argv[1] if len(sys.argv) > 1 else "What is the difference between SSH and TLS?"
    print(answer_question(test_q))
