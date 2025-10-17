from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import os
import json
import asyncio
from typing import AsyncGenerator

API_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

app = FastAPI(title="Face POC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video and return server path (or URL)."""
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(save_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)
    return JSONResponse({"ok": True, "path": save_path, "filename": file.filename})


@app.post("/api/process/video")
async def process_video(path: str = Form(...), card_conf: float = Form(0.5), frame_skip: int = Form(5)):
    """
    Start processing video on server. For demo, run processing synchronously and return a summary.
    In production you'd run a background task/queue and return job id.
    """
    # TODO: call your processing function (backend.face_core.process_video_file)
    # For demo, return a fake summary after a short wait
    await asyncio.sleep(0.5)
    demo = {
        "video": os.path.basename(path),
        "detections": 12,
        "cards": {"red": 2, "yellow": 1, "green": 0},
        "people": ["Alex Turner", "Unknown"],
    }
    return JSONResponse({"ok": True, "summary": demo})


@app.post("/api/process/images")
async def process_images(files: list[UploadFile] = File(...)):
    """Accept multiple images, run server processing and return recognized/unrecognized lists."""
    saved = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                await out.write(chunk)
        saved.append(dest)

    # TODO: replace with real processing call
    demo = {
        "recognized": [{"id": "img-rec-1", "name": "Alex", "thumb": "/placeholder.jpg"}],
        "unrecognized": [{"id": "img-un-1", "name": "Unknown", "thumb": "/placeholder.jpg"}],
    }
    return JSONResponse({"ok": True, "results": demo})


@app.post("/api/load-yolo")
async def load_yolo(file: UploadFile = File(...)):
    """Upload YOLO model file (.pt) and load it on server (async background)."""
    path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)
    # TODO: load model in background
    return JSONResponse({"ok": True, "path": path})


@app.get("/api/dashboard/stream")
async def dashboard_stream():
    """
    Server-Sent Events endpoint emitting JSON events with live dashboard updates.
    The generator yields 'data: <json>\\n\\n' chunks.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        # replace this demo loop with integration into your processing pipeline:
        for i in range(30):
            data = {"video": "demo.mp4", "time_idx": i, "stats": {"detections": i, "red": i//5}}
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1.0)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")