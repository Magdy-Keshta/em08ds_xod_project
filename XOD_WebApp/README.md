# XOD — X-ray Object Detection

A web console for X-ray baggage/parcel screening. Upload an X-ray, a trained
YOLO model flags objects with confidence scores and bounding boxes, and the
operator can push the image (brightness, contrast, gamma, negative, sharpen,
edge, false-colour, histogram equalise) to inspect it — all in the browser.

> Screening **aid** / decision support. It surfaces candidates for a human to
> verify; it is not a certified inspection device.

---

## How it's split

The server does one thing: run the model. Everything visual happens client-side.

- **Server (`app.py`, Flask + Ultralytics):** loads the `.pt` model, runs
  inference, returns JSON — class, confidence, and bbox in original-image
  pixels — plus the model's own timing breakdown. It never renders the image or
  draws boxes, so it never calls `results.plot()` (which would try to fetch
  fonts) and never touches the pixels beyond decode + optional downscale.
- **Browser (`static/js/app.js`):** decodes and displays the X-ray on a
  pan/zoom canvas, runs all enhancement on an offscreen buffer, and draws the
  detection overlay. Boxes, per-class toggles, the confidence slider, and both
  exports are all local.

This keeps per-request server load to just the model, which matters on the
shared host — and it's why enhancement is instant regardless of server latency.

---

## Screening tools (all client-side)

- Brightness / contrast / gamma
- Negative (invert)
- Sharpen (unsharp 3×3)
- Edge detection (Sobel)
- Histogram equalisation (preserves hue)
- False-colour palettes: grayscale, hot, rainbow (jet), ice
- Live luminance histogram
- Pan/zoom (wheel to zoom at cursor, drag to pan, fit, keyboard `+ - 0 b`)
- Click a box or a list row to select it
- Export annotated PNG (full resolution) and raw detection JSON

Class colours are stable per class id, so the same threat type is always the
same colour across the legend, the list, and the boxes.

---

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | The console |
| GET | `/api/status` | Model state + class list + display defaults |
| POST | `/api/detect` | multipart `image` (+ optional `conf`, `iou`) → detections JSON |
| POST | `/api/model` | multipart `model` (`.pt`) → validate & activate (guarded by `XOD_ADMIN_KEY` if set) |
| GET | `/healthz` | Liveness |

`/api/detect` response:

```json
{
  "ok": true,
  "image": { "width": 1600, "height": 640 },
  "model": { "name": "best.pt", "num_classes": 12 },
  "processing_time_ms": { "preprocess": 2.1, "inference": 24.0, "postprocess": 1.6, "total": 27.7 },
  "wall_time_ms": 41.3,
  "num_detections": 3,
  "counts": { "knife": 2, "gun": 1 },
  "detections": [
    { "id": 0, "class": "knife", "class_id": 3, "confidence": 0.94, "bbox": [x1, y1, x2, y2] }
  ]
}
```

`processing_time_ms` is what the model reports; `wall_time_ms` is the full
server round trip including decode and box rescaling. Boxes are in original
image pixels, sorted strongest first.

The class list is read from the model at load time (`model.names`) — nothing is
hardcoded, so any YOLO `.pt` with any number of classes works. Drop in your
12-class threat model and the legend, colours, and toggles populate themselves.

---

## Configuration

Everything is environment-overridable — see the table in **DEPLOY.md**. The
ones you'll actually touch: `XOD_MODEL` (weights filename), `XOD_ADMIN_KEY`
(lock model upload), `XOD_IMGSZ` (recall vs. cost).

---

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip uninstall -y opencv-python && pip install opencv-python-headless   # headless swap
# put weights at models/best.pt
python app.py            # http://localhost:5000
```

Production runs under Passenger via `passenger_wsgi.py` — see **DEPLOY.md**.

---

## Deploying

See **DEPLOY.md** for the full DirectAdmin walkthrough (`xod.tasknova.app`),
the OpenCV-headless swap, the two log files to watch, the torch memory/thread
notes, and the ONNX fallback if you outgrow the LVE memory cap.
