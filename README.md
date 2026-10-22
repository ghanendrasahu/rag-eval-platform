# RAG Platform API

Production-ready backend centered on one use case: PDF RAG with quality evaluation.

## Services

- RAG Question Answering API
- RAG Evaluation API (retrieval + answer grounding checks)

## Clean Folder Layout

Active project code lives in:
- `api/`
- `core/`
- `services/`
- `eval/`
- `utils/`
- `data/`

Legacy/unwanted files were moved to:
- `_archive/legacy/`
- `_archive/cache/`

## Architecture

- `api/main.py` - app entrypoint and shared middleware
- `core/*` - retrieval/generation pipeline orchestration
- `services/*` - vector store, embeddings, LLM integration
- `eval/*` - evaluation endpoints and scoring logic
- `utils/*` - request context and structured logging

## Local Run

```powershell
cd "C:\Users\ghane\Documents\ALL PROJECTS\DATA SCIENCE\RAG_EVAL_PLATFORM"
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## One-Time Cleanup (optional)

If old churn or deploy artifacts are still present, run:

```powershell
\.\\RAG_EVAL_PLATFORM_cleanup.ps1
```

## RAG Endpoints

- `GET /health`
- `POST /ingest`
- `POST /query`

## Evaluation Endpoints

- `GET /eval/health`
- `POST /eval/retrieval`
- `POST /eval/answer`

Sample `POST /eval/retrieval` body:

```json
{
  "question": "What does the document say about model evaluation?",
  "expected_sources": ["data/sample.pdf"],
  "top_k": 3
}
```

Sample `POST /eval/answer` body:

```json
{
  "question": "Summarize the main idea of the document.",
  "top_k": 3
}
```

## Logging

- Application logs: `logs.jsonl`

## Docker

```bash
docker build -t rag-platform-api:latest .
docker run --rm -p 8000:8000 --env-file .env rag-platform-api:latest
```

## License

MIT (c) 2026 ghanendrasahu - see [LICENSE](LICENSE).

Issues and pull requests are welcome.
