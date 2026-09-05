"""FastAPI entry point for LinePulse AI."""

from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import text

from linepulse.database.connection import engine


app = FastAPI(
    title="LinePulse AI API",
    version="0.1.0",
    description="Production risk intelligence API for LinePulse AI.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "LinePulse AI",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/database")
def database_health() -> dict[str, str]:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }
