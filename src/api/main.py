import uvicorn
from fastapi import FastAPI
from src.api.endpoints.ingestion.router import router as ingestion_router

app = FastAPI(title="Contract Review API", version="1.0.0")
app.include_router(ingestion_router, prefix="/api/v1") 

def main_entry() -> None:
    uvicorn.run(app, host="localhost", port=8000)

if __name__ == "__main__":
    main_entry() 