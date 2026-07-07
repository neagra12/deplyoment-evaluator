"""
FastAPI application entry point.
Run with: uvicorn main:app --reload --port 8000
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from routers import ingest, qa, eval, dashboard

app = FastAPI(
    title="Prox Grounded Product Expert",
    description="AI QA engine grounded in equipment shop manuals, with eval harness and rollout simulator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(qa.router)
app.include_router(eval.router)
app.include_router(dashboard.router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}
