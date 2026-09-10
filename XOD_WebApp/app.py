"""
XOD - X-ray Object Detection web app (Flask / Ultralytics).

The server does exactly one heavy thing: run the YOLO model. All image
enhancement (brightness, contrast, edges, false-colour, histogram) happens
in the browser, and bounding boxes are drawn client-side, so the server
never calls results.plot() (which would try to download fonts) and never
does per-request image processing.
"""
import os

# ---------------------------------------------------------------------------
# Thread hardening MUST happen before numpy / torch / cv2 are imported.
# On CloudLinux/LVE shared hosting the physical host reports many cores;
# BLAS/OpenMP will otherwise try to spawn a thread per core and blow past the
# account's thread quota (EAGAIN -> import crash). Passenger sets these too,
# but we repeat them so `python app.py` locally behaves identically.
# ---------------------------------------------------------------------------
for _var in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
# Keep Ultralytics fully offline and self-contained (no update pings, config
# and cache under our own data dir instead of a home directory that may be
# read-only on shared hosting).
os.environ.setdefault("YOLO_CONFIG_DIR", str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ultralytics")))
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

import io
import time
import json
import logging
import threading
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from config import Config

# PIL guard: allow large X-ray scans without the decompression-bomb warning
# turning into a hard error, but keep a sane ceiling.
Image.MAX_IMAGE_PIXELS = 200_000_000

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("xod")


# ---------------------------------------------------------------------------
# Model manager (lazy, thread-safe, hot-swappable)
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_model_meta = {
    "loaded": False,
    "name": None,
    "names": [],
    "num_classes": 0,
    "error": None,
}


def _disable_cv2_threads():
    """cv2 ships with Ultralytics; its thread pool hits the same LVE quota and
    can even starve Passenger's own fork(). setNumThreads(0) fully disables it
    (stronger than requesting 1). Safe no-op if cv2 isn't importable yet."""
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def _active_model_path():
    """Resolve which weights file to load: a previously uploaded/selected one
    (persisted across restarts) or the configured default."""
    try:
        if Config.ACTIVE_MODEL_FILE.exists():
            name = Config.ACTIVE_MODEL_FILE.read_text(encoding="utf-8").strip()
            if name:
                p = Config.MODEL_DIR / name
                if p.exists():
                    return p
    except Exception:
        pass
    default = Config.MODEL_DIR / Config.DEFAULT_MODEL_NAME
    return default if default.exists() else None


def _load_model(path):
    """Load YOLO weights and read class names. Raises on failure."""
    global _model, _model_meta
    from ultralytics import YOLO  # imports torch + cv2
    import torch
    torch.set_num_threads(1)
    _disable_cv2_threads()

    model = YOLO(str(path))
    # model.names is {index: label}; flatten to an ordered list.
    names_map = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = [names_map[i] for i in sorted(names_map.keys())]

    _model = model
    _model_meta = {
        "loaded": True,
        "name": Path(path).name,
        "names": names,
        "num_classes": len(names),
        "error": None,
    }
    log.info("Loaded model %s with %d classes", path.name, len(names))
    return model


def _ensure_model():
    """Return the loaded model, loading it on first use. Returns None if no
    model is configured or loading failed (meta carries the reason)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        path = _active_model_path()
        if path is None:
            _model_meta.update(loaded=False,
                               error="No model configured. Upload a .pt file or place one in models/.")
            return None
        try:
            return _load_model(path)
        except Exception as e:  # keep the app alive; surface the reason
            _model_meta.update(loaded=False, error="Failed to load model: %s" % e)
            log.exception("Model load failed")
            return None


# ---------------------------------------------------------------------------
# Image handling + inference
# ---------------------------------------------------------------------------
def _prepare_image(file_stream):
    """Open an upload, respect EXIF orientation, convert to RGB, and produce a
    (possibly downscaled) copy for inference. Returns
    (display_size, infer_image, scale) where scale maps infer-space -> original."""
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img)  # honour phone/camera rotation
    img = img.convert("RGB")
    ow, oh = img.size

    scale = 1.0
    infer_img = img
    longest = max(ow, oh)
    if Config.MAX_INFER_SIDE and longest > Config.MAX_INFER_SIDE:
        scale = Config.MAX_INFER_SIDE / float(longest)
        infer_img = img.resize(
            (max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))),
            Image.BILINEAR,
        )
    return (ow, oh), infer_img, scale


def _run_inference(model, infer_img, conf, iou):
    """Run YOLO and normalise the result. Box logic mirrors the reference
    Yolo-services.py (conf / cls / xyxy extraction)."""
    result = model.predict(
        infer_img,
        conf=conf,
        iou=iou,
        imgsz=Config.IMGSZ,
        verbose=False,
    )[0]

    # Timing reported by the model itself, in milliseconds.
    speed = getattr(result, "speed", {}) or {}
    pre = float(speed.get("preprocess", 0.0) or 0.0)
    inf = float(speed.get("inference", 0.0) or 0.0)
    post = float(speed.get("postprocess", 0.0) or 0.0)
    timing = {
        "preprocess": round(pre, 1),
        "inference": round(inf, 1),
        "postprocess": round(post, 1),
        "total": round(pre + inf + post, 1),
    }

    names_map = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        inv_scale = 1.0  # infer -> original mapping applied by caller
        for i in range(len(confs)):
            detections.append({
                "class_id": int(clss[i]),
                "class": names_map.get(int(clss[i]), str(int(clss[i]))),
                "confidence": float(confs[i]),
                "bbox": [float(v) for v in xyxy[i].tolist()],  # infer-space for now
            })
        _ = inv_scale
    return detections, timing


def _rescale_boxes(detections, scale, ow, oh):
    """Map boxes from inference-space back to the original image and clamp."""
    if scale and scale != 1.0:
        inv = 1.0 / scale
        for d in detections:
            d["bbox"] = [v * inv for v in d["bbox"]]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        x1 = max(0.0, min(x1, ow)); x2 = max(0.0, min(x2, ow))
        y1 = max(0.0, min(y1, oh)); y2 = max(0.0, min(y2, oh))
        d["bbox"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
    return detections


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.before_request
def _before():
    # Re-assert cv2 thread suppression on every request: Passenger's fork/spawn
    # model doesn't reliably preserve a module-level call.
    _disable_cv2_threads()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    """Model readiness + class list for the legend, plus display defaults."""
    _ensure_model()  # trigger a lazy load so the UI reflects real state
    return jsonify({
        "model": {
            "loaded": _model_meta["loaded"],
            "name": _model_meta["name"],
            "num_classes": _model_meta["num_classes"],
            "classes": _model_meta["names"],
            "error": _model_meta["error"],
        },
        "defaults": {
            "conf": Config.DEFAULT_CONF,
            "conf_floor": Config.CONF_FLOOR,
            "iou": Config.IOU,
            "imgsz": Config.IMGSZ,
        },
        "admin_locked": bool(Config.ADMIN_KEY),
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "image" not in request.files:
        return jsonify(ok=False, error="No image uploaded (field 'image')."), 400
    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify(ok=False, error="Empty filename."), 400
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in Config.ALLOWED_IMAGE_EXT:
        return jsonify(ok=False, error="Unsupported image type '%s'." % ext), 400

    model = _ensure_model()
    if model is None:
        return jsonify(ok=False, error=_model_meta.get("error") or "No model loaded."), 503

    # Per-request confidence override (defaults to the low server floor).
    try:
        conf = float(request.form.get("conf", Config.CONF_FLOOR))
    except (TypeError, ValueError):
        conf = Config.CONF_FLOOR
    conf = max(0.01, min(conf, 0.99))
    try:
        iou = float(request.form.get("iou", Config.IOU))
    except (TypeError, ValueError):
        iou = Config.IOU

    try:
        (ow, oh), infer_img, scale = _prepare_image(f.stream)
    except Exception as e:
        return jsonify(ok=False, error="Could not read image: %s" % e), 400

    t0 = time.perf_counter()
    detections, timing = _run_inference(model, infer_img, conf, iou)
    detections = _rescale_boxes(detections, scale, ow, oh)
    wall_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # Sort strongest first; give each a stable id for the UI.
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    for i, d in enumerate(detections):
        d["id"] = i
        d["confidence"] = round(d["confidence"], 4)

    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    return jsonify({
        "ok": True,
        "image": {"width": ow, "height": oh},
        "model": {"name": _model_meta["name"], "num_classes": _model_meta["num_classes"]},
        "processing_time_ms": timing,   # reported by the model
        "wall_time_ms": wall_ms,        # end-to-end incl. decode + rescale
        "num_detections": len(detections),
        "counts": counts,
        "detections": detections,
    })


@app.route("/api/model", methods=["POST"])
def api_model():
    """Upload/replace the active weights. Guarded by XOD_ADMIN_KEY if set."""
    if Config.ADMIN_KEY:
        key = request.headers.get("X-Admin-Key") or request.form.get("admin_key", "")
        if key != Config.ADMIN_KEY:
            return jsonify(ok=False, error="Invalid admin key."), 403

    if "model" not in request.files:
        return jsonify(ok=False, error="No model uploaded (field 'model')."), 400
    f = request.files["model"]
    if not f or f.filename == "":
        return jsonify(ok=False, error="Empty filename."), 400

    safe = secure_filename(f.filename)
    ext = Path(safe).suffix.lower()
    if ext not in Config.ALLOWED_MODEL_EXT:
        return jsonify(ok=False, error="Model must be a .pt file."), 400

    # Save to a temp name inside models/, validate it loads, then activate.
    dest = Config.MODEL_DIR / safe
    tmp = Config.MODEL_DIR / (safe + ".uploading")
    try:
        f.save(str(tmp))
    except Exception as e:
        return jsonify(ok=False, error="Could not save upload: %s" % e), 500

    global _model
    try:
        with _model_lock:
            #os.replace(str(tmp), str(dest))
            if dest.exists():
                dest.unlink()

            tmp.rename(dest)
            _model = None  # force reload from the new file
            _load_model(dest)
            Config.ACTIVE_MODEL_FILE.write_text(dest.name, encoding="utf-8")
    except Exception as e:
        # Roll back the active pointer/model reference on a bad file.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        _model = None
        return jsonify(ok=False, error="That file didn't load as a YOLO model: %s" % e), 400

    return jsonify({
        "ok": True,
        "model": {
            "name": _model_meta["name"],
            "num_classes": _model_meta["num_classes"],
            "classes": _model_meta["names"],
        },
    })


# ---------------------------------------------------------------------------
# Error handling: log full tracebacks server-side, return a generic message.
# HTTPExceptions (404 etc.) pass through untouched.
# ---------------------------------------------------------------------------
@app.errorhandler(Exception)
def _on_error(e):
    if isinstance(e, HTTPException):
        return e
    try:
        Config.ERROR_LOG.write_text(
            "%s\n\n%s" % (repr(e), traceback.format_exc()), encoding="utf-8"
        )
    except Exception:
        pass
    log.exception("Unhandled error")
    return jsonify(ok=False, error="Internal server error."), 500


if __name__ == "__main__":
    # Local dev server only; production runs under Passenger via passenger_wsgi.py
    app.run(host="0.0.0.0", port=5000, debug=True)
