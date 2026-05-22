from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from eval.service import evaluate_answer, evaluate_retrieval
from utils.logger import log_event

router = APIRouter(prefix="/eval", tags=["evaluation"])


class RetrievalEvalRequest(BaseModel):
    question: str = Field(min_length=1)
    expected_sources: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=20)


class AnswerEvalRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@router.get("/health")
def eval_health() -> dict:
    return {"status": "ok", "module": "rag-evaluation"}


@router.post("/retrieval")
def eval_retrieval(request: RetrievalEvalRequest) -> dict:
    try:
        result = evaluate_retrieval(
            question=request.question,
            expected_sources=request.expected_sources,
            top_k=request.top_k,
        )
        log_event("eval_retrieval", result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/answer")
def eval_answer(request: AnswerEvalRequest) -> dict:
    try:
        result = evaluate_answer(question=request.question, top_k=request.top_k)
        log_event("eval_answer", result)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

