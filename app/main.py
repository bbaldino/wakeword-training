from __future__ import annotations

import asyncio
import hashlib
import json
import os
import urllib.request
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


def _negatives_info() -> dict:
    """Best-effort count of flagged false positives available at the orchestrator."""
    url = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
    if not url:
        return {"url": "", "count": None, "reachable": False}
    try:
        data = urllib.request.urlopen(f"{url}/events?label=false", timeout=4).read()
        n = len(json.loads(data).get("events", []))
        return {"url": url, "count": n, "reachable": True}
    except Exception:
        return {"url": url, "count": None, "reachable": False}


# ── Pages ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = manager.state
    if state.status == TrainingStatus.RUNNING:
        return RedirectResponse("/status", status_code=303)
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "state": state,
        "negatives": _negatives_info(),
    })


@app.post("/start")
async def start_training(
    request: Request,
    wake_word: str = Form(...),
    n_samples: int = Form(10000),
    n_samples_val: int = Form(2000),
    training_steps: int = Form(50000),
    layer_size: int = Form(32),
    orchestrator_url: str = Form(""),
    include_negatives: bool = Form(False),
    push_model: bool = Form(False),
):
    params = TrainingParams(
        wake_word=wake_word,
        n_samples=n_samples,
        n_samples_val=n_samples_val,
        training_steps=training_steps,
        layer_size=layer_size,
        orchestrator_url=orchestrator_url,
        include_negatives=include_negatives,
        push_model=push_model,
    )
    try:
        manager.start_training(params)
    except RuntimeError as e:
        return templates.TemplateResponse(request, "index.html", {
            "request": request,
            "state": manager.state,
            "error": str(e),
            "negatives": _negatives_info(),
        }, status_code=409)
    return RedirectResponse("/status", status_code=303)


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    return templates.TemplateResponse(request, "status.html", {
        "request": request,
        "state": manager.state,
    })


@app.post("/cancel")
async def cancel_training():
    manager.cancel_training()
    return RedirectResponse("/status", status_code=303)


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _read_eval(stem: str):
    p = OUTPUT_DIR / f"{stem}.eval.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _list_output_models():
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
                    "sha256": _sha12(f) if f.suffix == ".onnx" else None,
                    "eval": _read_eval(f.stem) if f.suffix == ".onnx" else None,
                })
    return models


def _puck_models() -> dict:
    """Best-effort snapshot of what's currently deployed on the puck."""
    url = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
    if not url:
        return {"url": "", "reachable": False, "models": []}
    try:
        data = urllib.request.urlopen(f"{url}/models", timeout=4).read()
        return {"url": url, "reachable": True,
                "models": json.loads(data).get("models", [])}
    except Exception:
        return {"url": url, "reachable": False, "models": []}


def _models_context(request: Request, message: dict | None = None) -> dict:
    models = _list_output_models()
    puck = _puck_models()
    live = {m.get("sha256") for m in puck["models"] if m.get("sha256")}
    for m in models:
        m["live"] = bool(m.get("sha256")) and m["sha256"] in live
    return {"request": request, "models": models, "puck": puck, "message": message}


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    return templates.TemplateResponse(request, "models.html", _models_context(request))


def _puck_events() -> dict:
    """All collected wake events from the puck (each tagged with model_sha + ts)."""
    url = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
    if not url:
        return {"url": "", "reachable": False, "events": []}
    try:
        data = urllib.request.urlopen(f"{url}/events", timeout=8).read()
        d = json.loads(data)
        return {"url": url, "reachable": bool(d.get("enabled", True)),
                "events": d.get("events", [])}
    except Exception:
        return {"url": url, "reachable": False, "events": []}


def _fp_trend() -> dict:
    """Group wake events by the model version (`model_sha`) that fired them, so
    the real-world false-positive rate can be compared across builds over time."""
    puck = _puck_events()
    live = {m.get("sha256") for m in _puck_models()["models"] if m.get("sha256")}
    groups: dict[tuple, dict] = {}
    for e in puck["events"]:
        key = (e.get("model", "?"), e.get("model_sha") or "untagged")
        g = groups.setdefault(key, {
            "model": key[0], "sha": key[1],
            "total": 0, "false": 0, "real": 0, "unlabeled": 0,
            "first": None, "last": None,
        })
        g["total"] += 1
        lab = e.get("label", "unlabeled")
        g[lab if lab in ("false", "real") else "unlabeled"] += 1
        ts = e.get("ts") or ""
        if ts:
            g["first"] = ts if g["first"] is None else min(g["first"], ts)
            g["last"] = ts if g["last"] is None else max(g["last"], ts)
    rows = []
    for g in groups.values():
        days = None
        if g["first"] and g["last"]:
            try:
                span = datetime.fromisoformat(g["last"]) - datetime.fromisoformat(g["first"])
                days = span.total_seconds() / 86400.0
            except Exception:
                days = None
        reviewed = g["false"] + g["real"]
        g["days"] = days
        g["false_frac"] = (g["false"] / reviewed) if reviewed else None
        g["false_per_day"] = (g["false"] / days) if days and days >= 0.5 else None
        g["live"] = g["sha"] in live
        rows.append(g)
    rows.sort(key=lambda r: (r["first"] or ""))
    return {"puck": puck, "rows": rows}


@app.get("/trend", response_class=HTMLResponse)
async def trend_page(request: Request):
    ctx = _fp_trend()
    ctx["request"] = request
    return templates.TemplateResponse(request, "trend.html", ctx)


@app.post("/models/{filename}/deploy", response_class=HTMLResponse)
async def deploy_model(request: Request, filename: str):
    """Push an already-trained model to the orchestrator (post-hoc deploy)."""
    onnx = OUTPUT_DIR / filename
    url = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
    if (not filename.endswith(".onnx") or not onnx.exists()
            or onnx.resolve().parent != OUTPUT_DIR.resolve()):
        message = {"ok": False, "text": f"Cannot deploy {filename}."}
    elif not url:
        message = {"ok": False, "text": "ORCHESTRATOR_URL is not set on this container."}
    else:
        req = urllib.request.Request(
            f"{url}/models/{filename[:-5]}",
            data=onnx.read_bytes(),
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        token = os.environ.get("MODEL_PUSH_TOKEN", "")
        if token:
            req.add_header("X-Auth-Token", token)
        try:
            urllib.request.urlopen(req, timeout=30).read()
            message = {"ok": True,
                       "text": f"Deployed {filename} to {url} — hot-reloaded on the puck."}
        except Exception as e:
            message = {"ok": False, "text": f"Deploy failed: {e}"}
    return templates.TemplateResponse(request, "models.html", _models_context(request, message))


@app.post("/puck/models/{name}/rollback", response_class=HTMLResponse)
async def rollback_model(request: Request, name: str):
    """Ask the orchestrator to restore the backed-up model for {name}."""
    url = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
    if not url:
        message = {"ok": False, "text": "ORCHESTRATOR_URL is not set on this container."}
    else:
        req = urllib.request.Request(
            f"{url}/models/{name}/rollback", data=b"", method="POST"
        )
        token = os.environ.get("MODEL_PUSH_TOKEN", "")
        if token:
            req.add_header("X-Auth-Token", token)
        try:
            urllib.request.urlopen(req, timeout=30).read()
            message = {"ok": True,
                       "text": f"Rolled back '{name}' on the puck — hot-reloaded."}
        except Exception as e:
            message = {"ok": False, "text": f"Rollback failed: {e}"}
    return templates.TemplateResponse(request, "models.html", _models_context(request, message))


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
