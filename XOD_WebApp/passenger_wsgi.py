"""
Passenger entry point for DirectAdmin's "Setup Python App".

Order matters: the thread-limit environment variables have to be set BEFORE
numpy / torch / cv2 are imported anywhere, otherwise their BLAS/OpenMP layers
detect the physical host's full core count and try to spawn one thread per
core, blowing past the CloudLinux/LVE account quota (EAGAIN on import, or a
fork() failure that takes Passenger itself down with a bare Apache 500).
"""
import os
import sys

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Keep Ultralytics offline and pointed at a writable config/cache dir.
_here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_here, "data", "ultralytics"))
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")

sys.path.insert(0, _here)

from app import app as application  # noqa: E402  (import after env setup)

# Belt-and-braces: torch honours this at runtime too.
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
