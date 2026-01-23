import uvicorn
from fastapi import FastAPI

from src.api.endpoints.ingestion.router import router as ingestion_router
from src.config.logging import setup_logging
from src.config.settings import get_settings

setup_logging()
settings = get_settings()

app = FastAPI(title="Contract Review API", version="1.0.0", debug=settings.debug)
app.include_router(ingestion_router, prefix="/api/v1")


def main_entry() -> None:
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main_entry()
