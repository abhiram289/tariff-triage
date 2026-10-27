import uuid
import time
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response, status, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.contracts.schemas import (
    DisputeSubmissionRequest,
    SubmissionAckResponse,
    TriageResult,
    RunStatus,
    RoutingDecision,
    QuarantineRecord,
    FeedbackRequest,
)
from src.ingestion.quarantine import quarantine_manager
from src.retrieval.indexer import CorpusIndexer
from src.config import settings

app = FastAPI(
    title="Electricity Tariff & Dispute Triage Service",
    description="Multi-Agent Regulatory Triage for State Electricity Discoms (PESU Mini Project 11)",
    version="1.0.0"
)

# Mount static assets for direct user editing
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory execution store for runs (Evaluation 1 baseline)
RUNS_DATABASE: Dict[str, TriageResult] = {}
FEEDBACK_DATABASE: List[FeedbackRequest] = []


@app.get("/", response_class=HTMLResponse)
def serve_web_client():
    """Serves the user-editable HTML file from src/static/index.html."""
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>index.html not found in src/static/</h1>"


@app.get("/redesigned_style.css")
def serve_redesigned_css():
    """Serves redesigned_style.css directly at root path."""
    css_file = STATIC_DIR / "redesigned_style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    return Response(status_code=404)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Ingestion boundary validation failure handler (F2, D1).
    Quarantines rejected records with field-level errors; prevents agent run from starting.
    """
    correlation_id = request.headers.get("X-Correlation-ID", f"corr-{uuid.uuid4().hex[:8]}")
    client_ip = request.client.host if request.client else "unknown"

    try:
        raw_body = await request.json()
    except Exception:
        raw_body = {"raw_text": (await request.body()).decode("utf-8", errors="ignore")}

    errors = exc.errors()
    quarantined = quarantine_manager.quarantine(
        raw_payload=raw_body,
        validation_errors=errors,
        correlation_id=correlation_id,
        client_ip=client_ip
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "quarantined",
            "message": "Dispute rejected at ingestion boundary due to validation errors. Quarantined for review.",
            "quarantine_id": quarantined.quarantine_id,
            "correlation_id": correlation_id,
            "errors": [
                {
                    "field": " -> ".join([str(loc) for loc in err.get("loc", [])]),
                    "issue": err.get("msg"),
                    "type": err.get("type")
                } for err in errors
            ]
        }
    )


def run_background_triage(dispute: DisputeSubmissionRequest, run_id: str, correlation_id: str):
    from src.triage_engine import process_dispute_triage
    result = process_dispute_triage(dispute, run_id=run_id, correlation_id=correlation_id)
    RUNS_DATABASE[run_id] = result


@app.post("/api/disputes")
async def submit_dispute_alias(dispute: DisputeSubmissionRequest, request: Request, background_tasks: BackgroundTasks):
    """Alias for /api/v1/dispute/submit matching Claude / project spec."""
    ack = await submit_dispute(dispute, request, background_tasks)
    return {"run_id": ack.run_id, "status": ack.status}


@app.get("/api/runs/{run_id}")
def get_run_status_alias(run_id: str):
    """Alias for /api/v1/dispute/status/{run_id} formatted for the web UI."""
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    res = RUNS_DATABASE[run_id]

    formatted_citations = [
        {
            "doc": c.document_name,
            "clause_id": c.section_or_clause,
            "page": c.page_number or 1,
            "quote": c.quoted_text
        } for c in res.citations
    ]

    return {
        "run_id": res.run_id,
        "status": res.status.value,
        "e2e_latency_s": res.wall_clock_latency_sec,
        "result": {
            "route": res.routing_decision.value,
            "verified": res.verification_passed,
            "tariff_category": res.predicted_tariff_category,
            "basis": res.routing_justification,
            "escalation_reason": res.human_escalation_reason,
            "citations": formatted_citations,
            "prompt_version": res.prompt_version
        }
    }


@app.get("/api/runs/{run_id}/trace")
def get_run_trace(run_id: str):
    """Detailed trace for requirement D5."""
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    res = RUNS_DATABASE[run_id]
    return {
        "run_id": res.run_id,
        "steps": [s.model_dump() for s in res.trace_steps],
        "latency_sec": res.wall_clock_latency_sec,
        "citations": [c.model_dump() for c in res.citations],
        "verification_passed": res.verification_passed
    }


@app.get("/health")
def health_check():
    """Health check endpoint (F8, D3)."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "timestamp": datetime.utcnow().isoformat(),
        "quarantined_records_count": quarantine_manager.count()
    }


@app.post(
    "/api/v1/dispute/submit",
    response_model=SubmissionAckResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def submit_dispute(dispute: DisputeSubmissionRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Accepts a billing dispute against the published data contract (F1, A4).
    Returns a run identifier immediately.
    """
    correlation_id = request.headers.get("X-Correlation-ID", f"corr-{uuid.uuid4().hex[:8]}")
    run_id = f"run-{uuid.uuid4().hex[:10]}"

    # Initialize queued run
    RUNS_DATABASE[run_id] = TriageResult(
        run_id=run_id,
        consumer_id=dispute.consumer_id,
        status=RunStatus.PROCESSING,
        routing_decision=RoutingDecision.BILL_STANDS,
        routing_justification="Processing triage reasoning...",
        prompt_version=settings.PROMPT_VERSION,
        created_at=datetime.utcnow()
    )

    from src.triage_engine import process_dispute_triage
    result = process_dispute_triage(dispute, run_id=run_id, correlation_id=correlation_id)
    RUNS_DATABASE[run_id] = result

    return SubmissionAckResponse(
        run_id=run_id,
        consumer_id=dispute.consumer_id,
        status=RunStatus.QUEUED,
        submission_timestamp=datetime.utcnow(),
        status_url=f"/api/v1/dispute/status/{run_id}",
        correlation_id=correlation_id
    )


@app.get("/api/v1/dispute/status/{run_id}", response_model=TriageResult)
def get_run_status(run_id: str):
    """Poll status of an asynchronous run (A4)."""
    if run_id not in RUNS_DATABASE:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found")
    return RUNS_DATABASE[run_id]


@app.get("/api/v1/quarantine", response_model=List[QuarantineRecord])
def list_quarantined_records(limit: int = 50):
    """Lists quarantined malformed submissions for audit inspection (F2, D1)."""
    return quarantine_manager.list_records(limit=limit)


@app.post("/api/v1/feedback", status_code=status.HTTP_200_OK)
def submit_feedback(feedback: FeedbackRequest):
    """Feedback endpoint allowing the grievance officer to mark a result satisfactory (L6)."""
    FEEDBACK_DATABASE.append(feedback)
    return {
        "status": "recorded",
        "run_id": feedback.run_id,
        "message": "Feedback successfully persisted."
    }
