"""
FastAPI backend for RAG chatbot.
Endpoints: POST /upload, POST /ask, GET /metrics, GET /health, POST /eval, POST /verify.
Core logic in RAG_core.py with new modules for enhanced parsing, reranking, faithfulness.
"""
import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from RAG_core import RAGChatbotCore
from config import HOST, PORT, CHROMA_DIR, DATA_DIR, EVAL_DIR

logger = logging.getLogger(__name__)

# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Daily Prophet — Financial RAG API",
    description="PDF upload, Chroma vector store, financial analysis, risk detection, faithfulness verification",
    version="2.0.0"
)

# CORS — allow frontend dev server and all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global RAG instance ─────────────────────────────────────────────────
rag: Optional[RAGChatbotCore] = None


def get_rag() -> RAGChatbotCore:
    global rag
    if rag is None:
        rag = RAGChatbotCore()
    return rag


@app.on_event("startup")
async def startup_load():
    """Load persisted Chroma DB if it has data."""
    chroma_path = str(CHROMA_DIR)
    if os.path.exists(chroma_path):
        r = get_rag()
        embeddings_provider = os.getenv('EMBEDDINGS_PROVIDER', 'ollama')
        embeddings_model = os.getenv('EMBEDDINGS_MODEL', 'mxbai-embed-large')

        # Try Ollama first (requires ollama service running)
        if embeddings_provider == 'ollama':
            import socket
            ollama_available = False
            try:
                sock = socket.create_connection(('localhost', 11434), timeout=2)
                sock.close()
                ollama_available = True
            except (OSError, socket.timeout):
                logger.warning("Ollama not available at localhost:11434 — falling back to local embeddings")

            if not ollama_available:
                embeddings_provider = 'minimax_local'
                embeddings_model = 'sentence-transformers/all-MiniLM-L6-v2'

        r.config['EMBEDDINGS']['provider'] = embeddings_provider
        r.config['EMBEDDINGS']['model'] = embeddings_model
        try:
            r.load_persisted_vector_store(chroma_path)
            logger.info(f"Loaded persisted vector store from {chroma_path} (embeddings: {embeddings_provider})")
            qa_chain = r.create_qa_chain()
            if qa_chain:
                logger.info("QA chain ready for queries")
            else:
                logger.warning("QA chain could not be created")
        except Exception as e:
            logger.warning(f"Could not load persisted vector store: {e}")


# ── Request/Response models ────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    use_hyde: bool = False
    use_rerank: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list
    performance: dict
    from_cache: Optional[bool] = None
    error: Optional[str] = None
    financial_ratios: Optional[dict] = None
    risk_signals: Optional[dict] = None
    faithfulness: Optional[dict] = None


class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    tables_detected: Optional[int] = 0


class MetricsResponse(BaseModel):
    summary: dict
    metrics: list


# ── Endpoints ───────────────────────────────────────────────────────────
@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)):
    """Upload one or more PDF files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_paths = []
        for upload_file in files:
            if not upload_file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {upload_file.filename} is not a PDF"
                )
            tmp_path = os.path.join(tmpdir, upload_file.filename)
            content = await upload_file.read()
            with open(tmp_path, 'wb') as f:
                f.write(content)
            file_paths.append(tmp_path)
            logger.info(f"Saved uploaded file: {upload_file.filename}")

        r = get_rag()
        documents = r.load_pdfs(file_paths)
        if not documents:
            raise HTTPException(status_code=422, detail="No documents could be loaded from PDFs")

        vector_store = r.create_vector_store(documents)
        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to create vector store")

        chunks_count = len(r.document_splits)
        tables_detected = sum(1 for d in documents if d.metadata.get('type') == 'table')

        qa_chain = r.create_qa_chain()
        if not qa_chain:
            raise HTTPException(status_code=500, detail="Failed to create QA chain")

        return UploadResponse(
            message=f"Successfully processed {len(files)} file(s)",
            files_processed=len(files),
            chunks_created=chunks_count,
            tables_detected=tables_detected,
        )


@app.post("/ingest/ticker/{ticker}")
async def ingest_ticker(ticker: str):
    """Auto-fetch latest 10-K from SEC EDGAR by ticker and ingest."""
    try:
        from sec_edgar import fetch_10k
    except ImportError:
        raise HTTPException(status_code=500, detail="sec_edgar module not available")

    result = fetch_10k(ticker, output_dir=str(DATA_DIR))
    if not result:
        raise HTTPException(status_code=404, detail=f"Could not fetch 10-K for {ticker}")

    local_path = result["local_path"]
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail=f"Downloaded filing not found at {local_path}")

    r = get_rag()
    documents = r.load_pdfs([local_path])
    if not documents:
        raise HTTPException(status_code=422, detail="Could not load filing from SEC")

    # Use the existing vector store's embeddings to add new documents
    # (must match the original embeddings provider used when creating the store)
    store = r.create_vector_store(documents)
    if not store and not r.vector_store:
        raise HTTPException(status_code=500, detail="Failed to create vector store")

    # Re-create QA chain to include new docs
    qa_chain = r.create_qa_chain()
    if not qa_chain:
        raise HTTPException(status_code=500, detail="Failed to create QA chain")

    return {
        "message": f"Successfully ingested {ticker} 10-K ({result['year']})",
        "company": result["company"],
        "ticker": ticker.upper(),
        "year": result["year"],
        "filing_date": result["filing_date"],
        "chunks_created": len(r.document_splits),
        "local_path": local_path,
    }


@app.post("/ask/ticker")
async def ask_ticker(request: AskRequest):
    """Ask about a ticker: auto-fetches 10-K from SEC EDGAR and answers."""
    ticker = None
    if request.llm_base_url:
        ticker = request.llm_base_url.strip()

    if not ticker:
        # Try to extract ticker from the question
        words = request.question.strip().split()
        for w in words:
            w_clean = w.strip(".,!?:;'\"")
            if w_clean.isupper() and len(w_clean) <= 5 and w_clean.isalpha():
                ticker = w_clean
                break

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker not found. Pass it in llm_base_url field or include it in the question.")

    # Check if already ingested
    r = get_rag()
    if not r.qa_chain:
        # Auto-ingest
        try:
            from sec_edgar import fetch_10k
        except ImportError:
            raise HTTPException(status_code=500, detail="sec_edgar module not available")

        fetch_result = fetch_10k(ticker, output_dir=str(DATA_DIR))
        if not fetch_result:
            raise HTTPException(status_code=404, detail=f"Could not fetch 10-K for {ticker}")

        documents = r.load_pdfs([fetch_result["local_path"]])
        if not documents:
            raise HTTPException(status_code=422, detail="Could not load filing")

        r.create_vector_store(documents)
        r.create_qa_chain()

        if not r.qa_chain:
            raise HTTPException(status_code=500, detail="Failed to create QA chain")

    # Now answer the question
    result = r.ask_question(
        request.question,
        use_hyde=request.use_hyde,
        use_rerank=request.use_rerank,
    )

    if 'error' in result and not result.get('answer'):
        return AskResponse(
            answer="",
            sources=[],
            performance=result.get('performance', {}),
            error=result['error'],
        )

    return AskResponse(
        answer=result.get('answer', ''),
        sources=result.get('sources', []),
        performance=result.get('performance', {}),
        from_cache=result.get('from_cache', False),
        error=result.get('error'),
        financial_ratios=result.get('financial_ratios'),
        risk_signals=result.get('risk_signals'),
        faithfulness=result.get('faithfulness'),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a question against uploaded documents."""
    r = get_rag()

    if not r.qa_chain:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Please upload documents first via POST /upload."
        )

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.llm_provider or request.llm_model:
        if request.llm_provider:
            r.config['LLM']['provider'] = request.llm_provider
        qa_chain = r.create_qa_chain(
            llm_provider=request.llm_provider,
            llm_model=request.llm_model,
            system_prompt=request.system_prompt
        )
        if not qa_chain:
            raise HTTPException(status_code=500, detail="Failed to create QA chain with specified provider")

    result = r.ask_question(
        request.question,
        use_hyde=request.use_hyde,
        use_rerank=request.use_rerank,
    )

    if 'error' in result and not result.get('answer'):
        return AskResponse(
            answer="",
            sources=[],
            performance=result.get('performance', {}),
            error=result['error'],
            financial_ratios=result.get('financial_ratios'),
            risk_signals=result.get('risk_signals'),
        )

    return AskResponse(
        answer=result.get('answer', ''),
        sources=result.get('sources', []),
        performance=result.get('performance', {}),
        from_cache=result.get('from_cache', False),
        error=result.get('error'),
        financial_ratios=result.get('financial_ratios'),
        risk_signals=result.get('risk_signals'),
        faithfulness=result.get('faithfulness'),
    )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Streaming version of /ask using Server-Sent Events (SSE)."""
    r = get_rag()

    if not r.vector_store:
        async def err_gen():
            yield "data: [ERROR] Vector store not initialized. Please upload documents first.\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    if not request.question or not request.question.strip():
        async def empty_gen():
            yield "data: [ERROR] Question cannot be empty.\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    if request.llm_provider:
        r.config['LLM']['provider'] = request.llm_provider
    if request.llm_model:
        r.config['LLM']['model'] = request.llm_model

    async def event_stream():
        try:
            for chunk in r.ask_question_stream(
                question=request.question,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                system_prompt=request.system_prompt,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/tables")
async def list_tables():
    """List all available financial tables for Text-to-SQL."""
    try:
        from table_store import get_table_schemas, count_tables
        schemas = get_table_schemas()
        return {"count": count_tables(), "tables": schemas}
    except ImportError:
        return {"count": 0, "tables": [], "error": "table_store module not available"}
    except Exception as e:
        return {"count": 0, "tables": [], "error": str(e)}


@app.post("/verify")
async def verify_faithfulness(request: AskRequest):
    """Run faithfulness verification on a question+answer pair."""
    r = get_rag()
    if not r.qa_chain:
        raise HTTPException(status_code=503, detail="QA chain not initialized")

    result = r.ask_question(request.question, return_all_details=True)
    return {
        "question": request.question,
        "answer": result.get("answer", ""),
        "faithfulness": result.get("faithfulness"),
        "sources": result.get("sources", []),
        "financial_ratios": result.get("financial_ratios"),
        "risk_signals": result.get("risk_signals"),
    }


@app.post("/ask/table")
async def ask_table(request: AskRequest):
    """Answer a question via Text-to-SQL against stored financial tables."""
    r = get_rag()

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = r.ask_table(request.question)

    if result.get("error"):
        return {"answer": "", "error": result["error"], "has_data": False, "auto_routed": result.get("auto_routed", False)}

    return {
        "answer": result.get("answer", ""),
        "sql": result.get("sql"),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "has_data": result.get("has_data", False),
        "total_rows": result.get("total_rows", 0),
        "error": result.get("error"),
        "auto_routed": result.get("auto_routed", False),
    }


@app.post("/eval/run")
async def run_evaluation(sample_size: int = 5):
    """Run a quick RAG evaluation on sample questions."""
    try:
        r = get_rag()
        if not r.qa_chain:
            return {"error": "QA chain not initialized", "suggestion": "Upload documents first"}

        from evaluation_pipeline import run_ragas_evaluation
        metrics = run_ragas_evaluation(
            rag_core=r,
            sample_size=sample_size,
        )
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Return performance metrics."""
    r = get_rag()
    return MetricsResponse(
        summary=r.get_performance_summary(),
        metrics=r.get_all_metrics()
    )


@app.get("/")
async def root():
    return FileResponse("templates/index.html")


@app.get("/health")
async def health():
    """Health check."""
    r = get_rag()
    return {
        "status": "ok",
        "vector_store_ready": r.vector_store is not None,
        "qa_chain_ready": r.qa_chain is not None,
        "chunks_loaded": len(r.document_splits),
    }


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
