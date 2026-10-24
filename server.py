"""
server.py - Binzy, the Smart Waste Segregation Assistant (web app + API).

Run:   .venv\\Scripts\\python server.py            (http://localhost:8000)
       .venv\\Scripts\\python server.py --lan      (also reachable from a phone on the same Wi-Fi)

Images are decoded in memory, classified and discarded; nothing is written to disk.
"""
import argparse
import base64
import io
import json
import threading
from pathlib import Path

from waste_core import ARTIFACTS, CATEGORIES, CATEGORY_NAMES, CLASS_INFO, ROOT, apply_perturbations

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402

MAX_UPLOAD = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse decompression bombs

STATIC = ROOT / "static"
DEMO = ROOT / "demo_samples"
MEDIA_FILES = {
    "confusion_matrix_5.png": ARTIFACTS / "confusion_matrix_5.png",
    "confusion_matrix_12.png": ARTIFACTS / "confusion_matrix_12.png",
    "robustness_curves.png": ARTIFACTS / "robustness_curves.png",
    "architecture.png": ROOT / "deliverables" / "architecture.png",
}

app = FastAPI(title="Binzy - Smart Waste Segregation Assistant", docs_url=None, redoc_url=None)
_engine = None
_lock = threading.Lock()


def engine():
    global _engine
    if _engine is None:
        from hybrid import HybridClassifier
        _engine = HybridClassifier()
    return _engine


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.middleware("http")
async def _headers(request, call_next):
    resp = await call_next(request)
    # always revalidate the app shell so a restarted demo never shows stale JS/CSS
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.on_event("startup")
def _warm():
    engine()
    # first call compiles CUDA kernels; do it before the first user scan
    with _lock:
        engine().predict(Image.new("RGB", (224, 224), (128, 128, 128)))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    e = engine()
    demos = [{"file": p.name, "name": p.stem.split("_", 1)[-1].replace("_", " ")} for p in sorted(DEMO.glob("*.jpg"))]
    return {
        "categories": [{"name": c, **CATEGORIES[c],
                        "classes": [CLASS_INFO[k]["display"] for k in e.class_names if CLASS_INFO[k]["category"] == c]}
                       for c in CATEGORY_NAMES],
        "config": e.cfg,
        "demos": demos,
        "convnext_report": load_json(ARTIFACTS / "evaluation_report.json"),
        "hybrid_report": load_json(ARTIFACTS / "hybrid_report.json"),
        "split": load_json(ARTIFACTS / "split_summary.json"),
        "training_log": load_json(ARTIFACTS / "training_log.json"),
        "class_display": {k: v["display"] for k, v in CLASS_INFO.items()},
        "device": e.device,
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    gamma: float = Form(1.0),
    glare: float = Form(0.0),
    occlusion: float = Form(0.0),
    blur: int = Form(0),
    conf_threshold: float | None = Form(None),
    entropy_threshold: float | None = Form(None),
    ood_threshold: float | None = Form(None),
):
    data = await image.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image too large (max 15 MB)")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "This file is not a readable image")
    img.thumbnail((1024, 1024))

    gamma = min(max(gamma, 0.2), 3.0)
    glare = min(max(glare, 0.0), 1.0)
    occlusion = min(max(occlusion, 0.0), 80.0)
    blur = int(min(max(blur, 0), 12))
    perturbed = abs(gamma - 1) > 1e-3 or glare > 0 or occlusion > 0 or blur > 0
    if perturbed:
        img = apply_perturbations(img, gamma=gamma, glare=glare, occlusion_pct=occlusion, blur=blur)

    overrides = {k: v for k, v in {"conf_threshold": conf_threshold, "entropy_threshold": entropy_threshold,
                                   "ood_threshold": ood_threshold}.items() if v is not None}
    with _lock:
        res = engine().predict(img, overrides)

    info = CLASS_INFO[res["fine_class"]]
    res.update({
        "bin": CATEGORIES[res["category"]]["bin"],
        "color": CATEGORIES[res["category"]]["color"],
        "target": info["target"],
        "fine_display": info["display"],
        "instruction": info["instruction"],
    })
    if perturbed:
        buf = io.BytesIO()
        preview = img.copy()
        preview.thumbnail((640, 640))
        preview.save(buf, format="JPEG", quality=85)
        res["preview"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return JSONResponse(res)


@app.get("/media/{name}")
def media(name: str):
    if name in MEDIA_FILES and MEDIA_FILES[name].exists():
        return FileResponse(MEDIA_FILES[name])
    raise HTTPException(404)


@app.get("/gallery/{kind}/{name}")
def gallery(kind: str, name: str):
    if kind not in ("failures", "successes"):
        raise HTTPException(404)
    base = (ARTIFACTS / kind).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.exists() or path.suffix.lower() != ".jpg":
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/demo", StaticFiles(directory=DEMO), name="demo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces (phone on same Wi-Fi)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port, log_level="info")
