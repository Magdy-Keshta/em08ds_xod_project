"""
Configuration for the XOD (X-ray Object Detection) web app.

Everything is overridable via environment variables so you can tune the
deployed instance from the DirectAdmin Python App "Environment variables"
panel without editing code.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class Config:
    # --- paths -----------------------------------------------------------
    BASE_DIR = BASE_DIR
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOG_DIR = DATA_DIR / "logs"
    ACTIVE_MODEL_FILE = DATA_DIR / "active_model.txt"
    ERROR_LOG = LOG_DIR / "last_error.log"

    # The model file loaded at startup if no upload has replaced it.
    # Drop your trained weights in models/ and set XOD_MODEL to its name.
    DEFAULT_MODEL_NAME = os.environ.get("XOD_MODEL", "best.pt")

    # --- inference -------------------------------------------------------
    # The server returns every detection at/above CONF_FLOOR; the browser
    # then filters to whatever the operator picks on the slider. Keep the
    # floor low so low-confidence threats aren't silently dropped.
    CONF_FLOOR = _env_float("XOD_CONF_FLOOR", 0.10)
    DEFAULT_CONF = _env_float("XOD_DEFAULT_CONF", 0.25)
    IOU = _env_float("XOD_IOU", 0.45)
    IMGSZ = _env_int("XOD_IMGSZ", 640)  # raise to 1024/1280 for small-object recall
    # Downscale huge uploads before inference to protect the LVE memory cap.
    # Boxes are always scaled back to the original image, so this is invisible
    # to the operator.
    MAX_INFER_SIDE = _env_int("XOD_MAX_INFER_SIDE", 2048)

    # --- uploads ---------------------------------------------------------
    ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    ALLOWED_MODEL_EXT = {".pt"}
    MAX_CONTENT_LENGTH = _env_int("XOD_MAX_UPLOAD_MB", 256) * 1024 * 1024

    # --- admin -----------------------------------------------------------
    # Model swapping is the one privileged action. If this is empty the
    # upload endpoint is open (consistent with an open-access instance).
    # Set XOD_ADMIN_KEY to require the key in the admin panel.
    ADMIN_KEY = os.environ.get("XOD_ADMIN_KEY", "")


# Make sure runtime directories exist on import.
for _d in (Config.MODEL_DIR, Config.DATA_DIR, Config.LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
