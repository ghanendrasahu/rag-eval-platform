import re
from collections import Counter

from core.rag_pipeline import run_rag_pipeline
from services.vectorstore import retrieve_docs

WORD_RE = re.compile(r"\b[a-zA-Z0-9]+\b")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text or "")]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def evaluate_retrieval(question: str, expected_sources: list[str], top_k: int) -> dict:
    results = retrieve_docs(question, top_k=top_k)
    normalized_expected = {s.strip().lower() for s in expected_sources if s.strip()}
    retrieved_sources = [
        (item.get("metadata") or {}).get("source", "").strip().lower() for item in results
    ]

    matched = [src for src in retrieved_sources if src in normalized_expected]
    precision = _safe_ratio(len(matched), len(results))
    recall = _safe_ratio(len(set(matched)), len(normalized_expected))

    return {
        "question": question,
        "top_k": top_k,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "expected_sources_count": len(normalized_expected),
        "retrieved_sources_count": len(results),
        "retrieved_sources": retrieved_sources,
        "matched_sources": matched,
    }


def _top_claim_tokens(answer: str, max_tokens: int = 20) -> list[str]:
    tokens = [t for t in _tokenize(answer) if len(t) > 3]
    common = Counter(tokens).most_common(max_tokens)
    return [token for token, _ in common]


def evaluate_answer(question: str, top_k: int) -> dict:
    rag_output = run_rag_pipeline(question=question, top_k=top_k)
    answer = rag_output["answer"]
    sources = rag_output["sources"]
    context = " ".join((src.get("text") or "") for src in sources)

    answer_tokens = _tokenize(answer)
    context_tokens = set(_tokenize(context))
    overlapping = [token for token in answer_tokens if token in context_tokens]

    groundedness = _safe_ratio(len(overlapping), len(answer_tokens))
    faithfulness = groundedness

    unsupported = []
    for token in _top_claim_tokens(answer):
        if token not in context_tokens:
            unsupported.append(token)

    return {
        "question": question,
        "answer": answer,
        "top_k": top_k,
        "source_count": len(sources),
        "faithfulness_score": faithfulness,
        "groundedness_score": groundedness,
        "unsupported_claims": unsupported,
        "metadata": rag_output["metadata"],
    }

