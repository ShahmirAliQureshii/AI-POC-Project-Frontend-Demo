from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import os
import json
import asyncio
from typing import AsyncGenerator

# import face_core (new)
from backend import face_core

API_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

app = FastAPI(title="Face POC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ...existing code...
from fastapi import HTTPException

@app.post("/api/train/add")
async def api_add_known(name: str = Form(...), files: list[UploadFile] = File(...)):
    """
    Upload one or more images for a person and add/update their embedding.
    Returns JSON { ok: True, stored: N }.
    """
    if not name:
        raise HTTPException(status_code=400, detail="missing name")
    saved_paths = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                await out.write(chunk)
        saved_paths.append(dest)
    try:
        res = face_core.add_known_face_from_images(name, saved_paths)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
# ...existing code...



# NOTE: point BASE_DIR to the backend package folder so uploads match face_core paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# mount uploads directory so thumbnails returned by face_core (/uploads/thumbs/...) are reachable
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ...rest of file remains unchanged...
# (keep the endpoints as you already have)

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(save_path, "wb") as out:
        while content := await file.read(1024 * 1024):
            await out.write(content)
    return JSONResponse({"ok": True, "path": save_path, "filename": file.filename})


@app.post("/api/process/video")
async def process_video(path: str = Form(...), card_conf: float = Form(0.5), frame_skip: int = Form(5), max_frames: int = Form(0)):
    """
    Blocking (synchronous) processing that returns a summary.
    For long videos you may prefer using the streaming endpoint below.
    """
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "path not found"})
    max_frames = max_frames or None
    try:
        res = face_core.process_video_sync(path, card_conf=float(card_conf), frame_skip=int(frame_skip), max_frames=max_frames)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/process/images")
async def process_images(files: list[UploadFile] = File(...), detect_cards: bool = Form(False), card_conf: float = Form(0.5)):
    saved_paths = []
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        async with aiofiles.open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                await out.write(chunk)
        saved_paths.append(dest)
    try:
        result = face_core.process_images(saved_paths, detect_cards=bool(detect_cards), card_conf=float(card_conf))
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/load-yolo")
async def load_yolo(file: UploadFile = File(...)):
    # save model file
    dest = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)
    # attempt to load into face_core (blocking)
    try:
        face_core.load_models(yolo_path=dest)
        return JSONResponse({"ok": True, "path": dest})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/dashboard/stream")
async def dashboard_stream():
    """
    Example streaming endpoint that will run face_core.start_video_stream in background and stream queue items as SSE.
    For demo we use a small test video path; in production you'd connect this to a running job.
    """
    # For demo: nothing queued until a client calls process_video/upload; you can pass ?path=...
    # Accept an optional ?path query param
    async def event_generator() -> AsyncGenerator[str, None]:
        # use a demo video if none provided
        demo_video = os.path.join(UPLOAD_DIR, "demo.mp4")
        if not os.path.exists(demo_video):
            # send a few synthetic events
            for i in range(6):
                yield f"data: {json.dumps({'video': 'demo', 'time_idx': i, 'stats': {'detections': i}})}\n\n"
                await asyncio.sleep(1.0)
            return
        # start thread-based queue
        q = face_core.start_video_stream(demo_video, card_conf=0.5, frame_skip=5, max_frames=60)
        import queue
        while True:
            try:
                item = q.get(timeout=20)
            except queue.Empty:
                break
            # SSE framing
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("event") == "done":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")