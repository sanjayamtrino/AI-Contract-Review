"""
FastAPI application entry point.

Sets up middleware, registers routers, and manages the application lifecycle.
"""

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request

from src.api.context import clear_context, set_document_id, set_request_id, set_session_id
from src.api.endpoints.admin.router import router as admin_router
from src.api.endpoints.doc_information.router import router as doc_information_router
from src.api.endpoints.ingestion.router import router as ingestion_router
from src.api.endpoints.orchestrator.router import router as orchestrator_router
from src.api.endpoints.compare.router import router as compare_router
from src.api.endpoints.missing_clauses.router import router as missing_clauses_router
from src.api.endpoints.retrieval.router import router as retrieval_router
from src.api.endpoints.general_review.router import router as general_review_router
from src.api.endpoints.playbook_review.router import router as playbook_review_router
from src.api.endpoints.describe_draft.router import router as describe_draft_router
from src.config.logging import setup_logging
from src.config.settings import get_settings
from src.dependencies import initialize_dependencies, shutdown_dependencies

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""
    await initialize_dependencies()
    yield
    await shutdown_dependencies()


app = FastAPI(
    title="Contract Review API",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)


@app.middleware("http")
async def add_context_middleware(request: Request, call_next):
    """Propagate session/document/request IDs from headers into context."""
    set_session_id(request.headers.get("X-Session-ID"))
    set_document_id(request.headers.get("X-Document-ID"))
    set_request_id(request.headers.get("X-Request-ID"))

    try:
        response = await call_next(request)
    finally:
        clear_context()

    return response


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measure and return request processing time in response header."""
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response


# Register routers
app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1/chat")
app.include_router(admin_router, prefix="/api/v1/admin")
app.include_router(orchestrator_router, prefix="/api/v1/orchesrator")
app.include_router(doc_information_router, prefix="/api/v1/DocInfo")
app.include_router(missing_clauses_router, prefix="/api/v1/missing-clauses")
app.include_router(compare_router, prefix="/api/v1/compare")
app.include_router(general_review_router, prefix="/api/v1/general-review")
app.include_router(playbook_review_router, prefix="/api/v1/playbook-review")
app.include_router(describe_draft_router, prefix="/api/v1/describe-draft")


def main_entry() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main_entry()
