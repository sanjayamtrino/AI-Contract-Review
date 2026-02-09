import uvicorn
from fastapi import FastAPI

from src.api.endpoints.ingestion.router import router as ingestion_router
from src.api.endpoints.retrieval.router import router as retrieval_router
from src.config.logging import setup_logging
from src.config.settings import get_settings
from src.dependencies import initialize_dependencies, shutdown_dependencies

setup_logging()
settings = get_settings()

app = FastAPI(title="Contract Review API", version="1.0.0", debug=settings.debug)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services at application startup."""
    initialize_dependencies()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup services at application shutdown."""
    shutdown_dependencies()


app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")


def main_entry() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main_entry()
