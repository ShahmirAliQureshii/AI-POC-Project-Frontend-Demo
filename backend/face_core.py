# Minimal headless extraction of face-v1.py processing logic.
# This module is called from backend.server.main and must run without any GUI.

import os
import cv2
import json
import time
import threading
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from PIL import Image

# ...existing code...
import pickle
import numpy as np

def save_known_embeddings():
    """Persist current known_faces to disk."""
    emb_path = os.path.join(os.path.dirname(__file__), "known_embeddings.pkl")
    try:
        with open(emb_path, "wb") as f:
            pickle.dump(MODELS["known_faces"], f)
        logger.info("Saved known embeddings (%d entries) to %s", len(MODELS["known_faces"]), emb_path)
        return True
    except Exception as e:
        logger.warning("Failed to save known embeddings: %s", e)
        return False

def add_known_face_from_images(name: str, image_paths: List[str]) -> Dict[str, Any]:
    """
    Create/append embeddings for `name` from provided list of local image file paths.
    Returns { ok: True, stored: N } or error.
    """
    load_models()
    if MODELS.get("embedder") is None:
        raise RuntimeError("embedder (keras-facenet) not loaded. Install and restart backend.")

    embeddings = []
    for p in image_paths:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = []
            if MODELS["detector"]:
                try:
                    faces = MODELS["detector"].detect_faces(rgb)
                except Exception:
                    faces = []
            # prefer first detected face; otherwise use full image
            if faces:
                face = faces[0]
                x, y, w, h = face.get("box", (0, 0, 0, 0))
                x, y = max(0, x), max(0, y)
                x2 = min(x + max(0, w), rgb.shape[1] - 1)
                y2 = min(y + max(0, h), rgb.shape[0] - 1)
                crop = rgb[y:y2, x:x2]
            else:
                crop = rgb
            # resize to expected embedder input (FaceNet) and compute embedding
            try:
                resized = cv2.resize(crop, (160, 160))
                emb = MODELS["embedder"].embeddings([resized])[0]
                embeddings.append(np.array(emb, dtype=float))
            except Exception as e:
                logger.warning("failed to embed %s: %s", p, e)
                continue
        except Exception as e:
            logger.warning("error processing training image %s: %s", p, e)
    if not embeddings:
        return {"ok": False, "error": "no embeddings computed"}

    # average embeddings for this person
    mean_emb = np.mean(np.stack(embeddings), axis=0)
    MODELS["known_faces"][name] = mean_emb
    saved = save_known_embeddings()
    return {"ok": True, "stored": 1, "name": name, "saved": bool(saved)}
# ...existing code...

# Globals / lazy-loaded models
MODELS = {
    "detector": None,
    "embedder": None,
    "card_model": None,
    "card_names": {},
    "known_faces": {},
}

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")
THUMBS_DIR = os.path.join(UPLOADS_DIR, "thumbs")
os.makedirs(THUMBS_DIR, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading models: attempting mtcnn, keras-facenet, ultralytics...")

def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def load_models(yolo_path: Optional[str] = None):
    """Try to load detector/embedder/yolo. Safe to call multiple times."""
    # MTCNN detector
    if MODELS["detector"] is None:
        MTCNN = _safe_import("mtcnn")
        if MTCNN:
            try:
                MODELS["detector"] = MTCNN.MTCNN()
                logger.info("MTCNN loaded")
            except Exception as e:
                MODELS["detector"] = None
                logger.warning("MTCNN import succeeded but failed to initialize: %s", e)
        else:
            logger.info("mtcnn not available")

    # FaceNet embedder
    if MODELS["embedder"] is None:
        FaceNet = _safe_import("keras_facenet")
        if FaceNet:
            try:
                MODELS["embedder"] = FaceNet.FaceNet()
                logger.info("keras-facenet FaceNet loaded")
            except Exception as e:
                MODELS["embedder"] = None
                logger.warning("keras-facenet import succeeded but failed to initialize: %s", e)
        else:
            logger.info("keras-facenet not available")

    # YOLO (optional)
    if MODELS["card_model"] is None and yolo_path:
        try:
            ultralytics = _safe_import("ultralytics")
            if ultralytics:
                try:
                    MODELS["card_model"] = ultralytics.YOLO(yolo_path)
                    names = getattr(MODELS["card_model"], "names", {})
                    if isinstance(names, list):
                        names = {i: n for i, n in enumerate(names)}
                    MODELS["card_names"] = names or {}
                    logger.info("ultralytics YOLO model loaded from %s", yolo_path)
                except Exception as e:
                    MODELS["card_model"] = None
                    logger.warning("Failed to initialize YOLO model: %s", e)
            else:
                logger.info("ultralytics not available")
        except Exception as e:
            MODELS["card_model"] = None
            logger.warning("Error while loading YOLO model: %s", e)

    # try load known faces from saved pickles if present
    emb_path = os.path.join(os.path.dirname(__file__), "known_embeddings.pkl")
    try:
        import pickle
        if os.path.exists(emb_path):
            with open(emb_path, "rb") as f:
                MODELS["known_faces"] = pickle.load(f) or {}
                logger.info("Loaded known embeddings (%d entries)", len(MODELS["known_faces"]))
        else:
            logger.info("No known_embeddings.pkl found")
    except Exception as e:
        MODELS["known_faces"] = {}
        logger.warning("Failed to load known embeddings: %s", e)
# ...existing code...

def _thumb_for_image(img_bgr, out_name: str, maxsize=(400, 400)) -> str:
    """Save a JPEG thumbnail to uploads/thumbs and return relative URL path (/uploads/thumbs/...)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil.thumbnail(maxsize)
    out_path = os.path.join(THUMBS_DIR, out_name)
    pil.save(out_path, format="JPEG", quality=80)
    # serve under /uploads/thumbs/ from FastAPI static mount
    return f"/uploads/thumbs/{out_name}"


def classify_card_label(raw_name: Optional[str], cls_id=None) -> Optional[str]:
    if raw_name is None:
        raw_name = ""
    l = str(raw_name).lower()
    if "red" in l: return "red"
    if "yellow" in l: return "yellow"
    if "green" in l: return "green"
    if cls_id is not None:
        try:
            cid = int(cls_id)
            return {0: "red", 1: "yellow", 2: "green"}.get(cid)
        except Exception:
            pass
    return None


def process_images(paths: List[str], detect_cards: bool = False, card_conf: float = 0.5) -> Dict[str, Any]:
    """
    Process image files and return recognized / unrecognized lists.
    Each item: { id, name, thumb, score, emotion, filename }
    """
    load_models()
    detector = MODELS.get("detector")
    embedder = MODELS.get("embedder")
    card_model = MODELS.get("card_model")
    card_names = MODELS.get("card_names", {})

    recognized = []
    unrecognized = []

    for p in paths:
        try:
            frame = cv2.imread(p)
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # card detection (optional)
            if detect_cards and card_model is not None:
                try:
                    results = card_model.predict(frame, imgsz=640, conf=card_conf, verbose=False)
                    # not used here beyond drawing; keep minimal
                except Exception:
                    pass

            faces = []
            if detector is not None:
                try:
                    faces = detector.detect_faces(rgb)
                except Exception:
                    faces = []

            # If no detector, attempt simple full-image as one face
            if not faces:
                # still produce a thumbnail and mark unknown
                thumb_url = _thumb_for_image(frame, f"{os.path.basename(p)}.jpg")
                unrecognized.append({"id": os.path.basename(p), "name": "Unknown", "thumb": thumb_url, "score": None, "emotion": None, "filename": os.path.basename(p)})
                continue

            for i, face in enumerate(faces):
                x, y, w, h = face.get("box", (0, 0, 0, 0))
                x = max(0, x); y = max(0, y)
                x2 = min(x + max(0, w), rgb.shape[1] - 1)
                y2 = min(y + max(0, h), rgb.shape[0] - 1)
                face_crop = rgb[y:y2, x:x2]
                # embed/compare
                name = "Unknown"; score = None
                if embedder is not None:
                    try:
                        emb = embedder.embeddings([cv2.resize(face_crop, (160, 160))])[0]
                        min_d = float("inf")
                        for kn, kem in MODELS.get("known_faces", {}).items():
                            import numpy as np
                            d = float(np.linalg.norm(emb - kem))
                            if d < min_d and d < 0.9:
                                min_d = d; name = kn; score = d
                    except Exception:
                        pass
                # emotion via deepface if available
                emotion = None
                try:
                    DeepFace = _safe_import("deepface")
                    if DeepFace:
                        # deepface.analyze may be heavy; call defensively
                        res = DeepFace.DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                        if isinstance(res, list) and res:
                            emotion = res[0].get("dominant_emotion")
                        elif isinstance(res, dict):
                            emotion = res.get("dominant_emotion")
                except Exception:
                    emotion = None

                # create thumb for the whole image (not face-only) for UI consistency
                thumb = _thumb_for_image(frame, f"{os.path.basename(p)}_{i}.jpg")
                item = {"id": f"{os.path.basename(p)}_{i}", "name": name, "thumb": thumb, "score": score, "emotion": emotion, "filename": os.path.basename(p)}
                if name == "Unknown":
                    unrecognized.append(item)
                else:
                    recognized.append(item)
        except Exception as e:
            # continue with other files; keep errors visible in logs
            print("process_images error:", e)

    return {"ok": True, "recognized": recognized, "unrecognized": unrecognized}


def process_video_sync(video_path: str, card_conf: float = 0.5, frame_skip: int = 5, max_frames: Optional[int] = None) -> Dict[str, Any]:
    """
    Synchronous (blocking) video processing. Returns a summary dict.
    Use carefully (long-running). For streaming updates prefer the threaded start_video_stream().
    """
    load_models()
    detector = MODELS.get("detector")
    embedder = MODELS.get("embedder")
    card_model = MODELS.get("card_model")
    card_names = MODELS.get("card_names", {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = 0
    detections_total = 0
    per_person = defaultdict(lambda: {"detections": 0, "best_score": None, "emotions": Counter(), "red": 0, "yellow": 0, "green": 0})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cards
        if card_model is not None:
            try:
                results = card_model.predict(frame, imgsz=640, conf=card_conf, verbose=False)
                for r in results:
                    boxes = getattr(r.boxes, "xyxy", None)
                    classes = getattr(r.boxes, "cls", None)
                    confs = getattr(r.boxes, "conf", None)
                    if boxes is not None:
                        boxes = boxes.cpu().numpy()
                        classes = classes.cpu().numpy() if classes is not None else []
                        confs = confs.cpu().numpy() if confs is not None else []
                        for (x1, y1, x2, y2), cls, cs in zip(boxes, classes, confs):
                            kind = classify_card_label(card_names.get(int(cls), str(cls)) if card_names else str(cls), int(cls))
                            if kind:
                                per_person["Unknown"][kind] += 1
            except Exception:
                pass

        faces = []
        if detector is not None:
            try:
                faces = detector.detect_faces(rgb)
            except Exception:
                faces = []

        for face in faces:
            x, y, w, h = face.get("box", (0, 0, 0, 0))
            x = max(0, x); y = max(0, y)
            x2 = min(x + max(0, w), rgb.shape[1] - 1)
            y2 = min(y + max(0, h), rgb.shape[0] - 1)
            fc = rgb[y:y2, x:x2]
            name = "Unknown"
            score = None
            if embedder is not None:
                try:
                    emb = embedder.embeddings([cv2.resize(fc, (160, 160))])[0]
                    import numpy as np
                    min_d = float("inf")
                    for kn, kem in MODELS.get("known_faces", {}).items():
                        d = float(np.linalg.norm(emb - kem))
                        if d < min_d and d < 0.9:
                            min_d = d; name = kn; score = d
                except Exception:
                    pass
            detections_total += 1
            per_person[name]["detections"] += 1
        # optional max frames break (for faster testing)
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    summary = {"video": os.path.basename(video_path), "frames_seen": frame_count, "detections": detections_total, "people": list(per_person.keys()), "per_person": {k: dict(v) for k, v in per_person.items()}}
    return {"ok": True, "summary": summary}


# Threaded streaming helper: returns a queue the caller can read from
def start_video_stream(video_path: str, card_conf: float = 0.5, frame_skip: int = 5, max_frames: Optional[int] = None):
    """
    Start processing in a background thread and return a queue-like object with .get() that yields JSON-ready updates.
    We'll implement a simple thread + list buffer + done flag for synchronous server streaming.
    """
    from queue import Queue
    q = Queue()

    def worker():
        q.put({"event": "started", "video": os.path.basename(video_path)})
        load_models()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            q.put({"event": "error", "message": "cannot open video"})
            q.put({"event": "done"})
            return
        frame_count = 0
        detections_total = 0
        per_person = defaultdict(lambda: {"detections": 0, "best_score": None, "emotions": Counter(), "red": 0, "yellow": 0, "green": 0})
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                # face detection (light)
                try:
                    faces = MODELS["detector"].detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if MODELS["detector"] else []
                except Exception:
                    faces = []
                detections_total += len(faces)
                if frame_count % max(1, frame_skip*2) == 0:
                    # push a periodic update
                    q.put({"event": "progress", "frame": frame_count, "detections": detections_total, "time": time.time()})
                if max_frames and frame_count >= max_frames:
                    break
        except Exception as e:
            q.put({"event": "error", "message": str(e)})
        finally:
            cap.release()
            q.put({"event": "done", "summary": {"video": os.path.basename(video_path), "frames_seen": frame_count, "detections": detections_total}})
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return q