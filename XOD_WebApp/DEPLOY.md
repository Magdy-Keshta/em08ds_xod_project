# Deploying XOD to `xod.tasknova.app` (DirectAdmin + Passenger)

This mirrors the setup already running for the UFF viewer at
`xray.tasknova.app`. Same CloudLinux/LVE shared host, same Passenger/WSGI
integration — with the extra care that the PyTorch runtime needs.

---

## 1. Upload the app

Put the whole `XOD_WebApp/` folder somewhere in your home directory, **not**
inside `public_html`. For example:

```
/home/<user>/xod-webapp/
```

Drop your trained weights into `models/` (see step 5) — the folder is already
there with a placeholder note.

---

## 2. Create the Python application

DirectAdmin → **Setup Python App** → Create:

| Field | Value |
|---|---|
| Python version | 3.10 or 3.11 |
| Application root | `xod-webapp` (the folder from step 1) |
| Application URL | `xod.tasknova.app` |
| Application startup file | `passenger_wsgi.py` |
| Application entry point | `application` |

Save. DirectAdmin creates a dedicated virtualenv and writes its own
`passenger_wsgi.py`. **Keep the one shipped in this project** — it sets the
thread-limit environment variables *before* torch/numpy import, which is the
whole reason the xray app stays up. If DirectAdmin overwrote it, restore this
project's version.

> **App-root gotcha (same as last time):** the path shown in the panel is
> relative to your home dir. If you see a bare Apache 500 with nothing in the
> Flask logs, 90% of the time the app root is pointing one level off and
> Passenger can't find `passenger_wsgi.py`.

---

## 3. Install dependencies

Copy the "Enter to the virtual environment" command from the panel, then:

```bash
source /home/<user>/virtualenv/xod-webapp/3.11/bin/activate && cd /home/<user>/xod-webapp
pip install --no-cache-dir -r requirements.txt
```

`--no-cache-dir` matters on shared hosting — the torch wheel is large and the
cache can trip the LVE disk/inode quota. If `pip` gets killed mid-install,
re-run it (it resumes) or install the heavy ones one at a time:

```bash
pip install --no-cache-dir torch torchvision
pip install --no-cache-dir ultralytics
```

---

## 4. Swap OpenCV for the headless build  ← don't skip this

Ultralytics depends on `opencv-python`, which needs system GL libraries
(`libGL.so.1`) that DirectAdmin hosts don't have. Left as-is you'll get
`ImportError: libGL.so.1: cannot open shared object file` on the first detect.
Replace it with the headless build:

```bash
pip uninstall -y opencv-python
pip install --no-cache-dir opencv-python-headless
```

(The app itself never calls OpenCV, but Ultralytics imports it on load.)

---

## 5. Add your model

Put the trained weights in `models/`:

```
/home/<user>/xod-webapp/models/best.pt
```

- If the file is named something else, set `XOD_MODEL=yourname.pt` (step 6).
- Files over ~100 MB: upload with **SFTP** straight into `models/` rather than
  the browser uploader — shared-host request buffers and Passenger timeouts
  make large HTTP uploads flaky. The in-app uploader is best for smaller `.pt`
  files and quick swaps.
- The model loads lazily on the first request, so the app starts fine even
  before the file is in place; `/api/status` will just report "no model".

---

## 6. Environment variables (panel → "Environment variables")

All optional — the defaults are sensible. The thread limits are already forced
in `passenger_wsgi.py`, so you don't need to set those by hand.

| Variable | Default | Purpose |
|---|---|---|
| `XOD_MODEL` | `best.pt` | Weights filename in `models/` |
| `XOD_ADMIN_KEY` | *(empty)* | Set a value to require a key for model upload. Empty = open, matching an open-access instance. |
| `XOD_IMGSZ` | `640` | Raise to `1024`/`1280` for better small-object recall (more RAM + time) |
| `XOD_DEFAULT_CONF` | `0.25` | Slider's starting confidence |
| `XOD_CONF_FLOOR` | `0.10` | Lowest confidence the server returns; the browser filters upward |
| `XOD_MAX_INFER_SIDE` | `2048` | Huge uploads are downscaled to this before inference (boxes are scaled back) |
| `XOD_MAX_UPLOAD_MB` | `256` | Max request size |

---

## 7. Restart & test

Hit **Restart** in the panel (or `touch tmp/restart.txt`), then visit
`https://xod.tasknova.app`.

Quick check without the browser:

```bash
curl -s https://xod.tasknova.app/api/status | python -m json.tool
```

You want `"loaded": true` and your class list. Then upload an X-ray in the UI —
you should get boxes, per-class confidences, and the model's own timing
breakdown.

---

## Diagnostics (two logs, same as xray)

1. **`stderr.log`** in the app root — Passenger/native crashes (thread quota,
   missing `libGL`, torch import failures). This is where a bare 500 explains
   itself.
2. **`data/logs/last_error.log`** — Python tracebacks from inside the app
   (caught by the global error handler).

Common ones:

| Symptom | Cause | Fix |
|---|---|---|
| Bare Apache 500, nothing in Flask logs | thread quota on import, or app-root off | confirm `passenger_wsgi.py` is this project's; check app root |
| `libGL.so.1: cannot open shared object file` | non-headless OpenCV | step 4 |
| `/api/status` → `"loaded": false` with an error | bad path or torch not installed | check `models/best.pt`, re-run install |
| 503 on detect after a while | LVE out-of-memory kill | smaller model / lower `XOD_IMGSZ`, or ONNX (below) |

---

## If you hit the memory wall — ONNX escape hatch

PyTorch is the heavy part. A resident torch process plus a mid-size YOLO model
can sit at several hundred MB, and under the account's LVE memory cap a large
model or concurrent requests can get OOM-killed (shows up as intermittent 503s
and an entry in `stderr.log`). Two levers first: use a smaller variant
(`yolov8n`/`s`) and keep `XOD_IMGSZ` at 640.

If that isn't enough, drop torch entirely and serve the model with ONNX
Runtime, which uses a fraction of the memory and a ~50 MB dependency instead of
torch's hundreds of MB.

On your Mac (where torch works fine):

```bash
pip install ultralytics onnx onnxruntime
yolo export model=best.pt format=onnx imgsz=640    # -> best.onnx
```

Then the server swaps `ultralytics`/`torch` for `onnxruntime` and does the
letterbox + NMS in NumPy. That's a self-contained change to the model manager
in `app.py` (the routes, the UI, and the JSON contract stay identical). I left
the current app on the PyTorch path you picked; if you want the ONNX-serving
variant, it's a drop-in replacement for `app.py` — say the word and I'll hand
it over.
