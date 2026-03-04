from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .models import TrainingParams, TrainingStatus
from .training import OUTPUT_DIR, manager

app = FastAPI(title="Wake Word Training")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ── Pages ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = manager.state
    if state.status == TrainingStatus.RUNNING:
        return RedirectResponse("/status", status_code=303)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "state": state,
    })


@app.post("/start")
async def start_training(
    request: Request,
    wake_word: str = Form(...),
    n_samples: int = Form(10000),
    n_samples_val: int = Form(2000),
    training_steps: int = Form(50000),
    layer_size: int = Form(32),
):
    params = TrainingParams(
        wake_word=wake_word,
        n_samples=n_samples,
        n_samples_val=n_samples_val,
        training_steps=training_steps,
        layer_size=layer_size,
    )
    try:
        manager.start_training(params)
    except RuntimeError as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "state": manager.state,
            "error": str(e),
        }, status_code=409)
    return RedirectResponse("/status", status_code=303)


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    return templates.TemplateResponse("status.html", {
        "request": request,
        "state": manager.state,
    })


@app.post("/cancel")
async def cancel_training():
    manager.cancel_training()
    return RedirectResponse("/status", status_code=303)


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    models = []
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.suffix in (".onnx", ".tflite"):
                stat = f.stat()
                models.append({
                    "name": f.name,
                    "size": _format_size(stat.st_size),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                })
    return templates.TemplateResponse("models.html", {
        "request": request,
        "models": models,
    })


@app.get("/models/{filename}")
async def download_model(filename: str):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        return JSONResponse({"error": "File not found"}, status_code=404)
    # Prevent path traversal
    if filepath.resolve().parent != OUTPUT_DIR.resolve():
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    return FileResponse(filepath, filename=filename)


# ── API ──────────────────────────────────────────────────────────────────────


@app.get("/api/state")
async def api_state():
    return manager.state.model_dump()


@app.get("/api/logs/stream")
async def logs_stream(request: Request):
    queue = manager.subscribe()

    async def event_generator():
        # Send existing buffer
        for line in manager.get_log_lines():
            yield {"event": "log", "data": line}

        # Stream new lines
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": "log", "data": line}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        finally:
            manager.unsubscribe(queue)

    return EventSourceResponse(event_generator())


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
