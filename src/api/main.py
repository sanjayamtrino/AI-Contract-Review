from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.endpoints.admin.router import router as admin_router
from src.api.endpoints.ingestion.router import router as ingestion_router
from src.api.endpoints.retrieval.router import router as retrieval_router
from src.config.logging import setup_logging
from src.config.settings import get_settings
from src.dependencies import initialize_dependencies, shutdown_dependencies

setup_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_dependencies()
    yield
    # Shutdown
    await shutdown_dependencies()


app = FastAPI(
    title="Contract Review API",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)


app.include_router(ingestion_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1/admin")


def main_entry() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main_entry()
