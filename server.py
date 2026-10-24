"""
server.py - Binzy, the Smart Waste Segregation Assistant (web app + API).

Run:   ./run_app.sh                                (macOS / Linux, http://localhost:8000)
       run_app.bat                                 (Windows)
       .venv/bin/python server.py --lan            (also reachable from a phone on same Wi-Fi)

Images are decoded in memory, classified and discarded; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from pathlib import Path
from typing import Optional

from waste_core import (
    ARTIFACTS,
    CATEGORIES,
    CATEGORY_NAMES,
    CLASS_INFO,
    DEMONSTRATION_NOTICE,
    MODEL_PATH,
    POLICY,
    POLICY_PATH,
    ROOT,
    UNCERTAIN_ADVICE,
    apply_perturbations,
    get_device,
)

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
_engine_error = None
_lock = threading.Lock()


def get_configured_engine(device_override: Optional[str] = None):
    global _engine, _engine_error
    if _engine is not None:
        return _engine

    if not MODEL_PATH.exists():
        _engine_error = (
            f"Mandatory checkpoint '{MODEL_PATH.name}' is missing in artifacts/. "
            "Inference is unavailable until the model checkpoint is transferred. "
            "Run 'python preflight.py' to inspect prerequisites."
        )
        return None

    try:
        from hybrid import HybridClassifier

        _engine = HybridClassifier(device=device_override)
        _engine_error = None
        return _engine
    except Exception as exc:
        _engine_error = f"Failed to initialize HybridClassifier: {type(exc).__name__}: {exc}"
        return None


def engine():
    e = get_configured_engine()
    if e is None:
        raise HTTPException(
            status_code=503,
            detail=(
                _engine_error
                or "Model service unavailable: required model artifacts are missing. "
                "Run 'python preflight.py' to view the transfer checklist."
            ),
        )
    return e


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
    e = get_configured_engine()
    if e is not None:
        try:
            with _lock:
                e.predict(Image.new("RGB", (224, 224), (128, 128, 128)))
            print(f"[READY] Binzy engine initialized successfully on device: {e.device}")
        except Exception as exc:
            print(f"[WARNING] Engine warmup encountered an issue: {exc}")
    else:
        print(f"[PREFLIGHT NOTICE] {_engine_error}")
        print("[PREFLIGHT NOTICE] Server running in diagnostic/preflight mode. Full inference is disabled until artifacts are provided.")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
@app.get("/api/ready")
def readiness():
    """Honest readiness and prerequisite health check endpoint."""
    missing = []
    if not MODEL_PATH.exists():
        missing.append("artifacts/model.pt")
    if not (ARTIFACTS / "siglip_probe.npz").exists():
        missing.append("artifacts/siglip_probe.npz")
    if not (ARTIFACTS / "hybrid_config.json").exists():
        missing.append("artifacts/hybrid_config.json")

    reports = {
        "evaluation_report": (ARTIFACTS / "evaluation_report.json").exists(),
        "hybrid_report": (ARTIFACTS / "hybrid_report.json").exists(),
        "split_summary": (ARTIFACTS / "split_summary.json").exists(),
        "training_log": (ARTIFACTS / "training_log.json").exists(),
    }

    is_ready = _engine is not None or (len(missing) == 0 and MODEL_PATH.exists())
    return {
        "status": "ready" if is_ready else "missing_artifacts",
        "inference_ready": is_ready,
        "device": _engine.device if _engine else get_device(),
        "components": {
            "convnext_model": MODEL_PATH.exists(),
            "siglip_probe": (ARTIFACTS / "siglip_probe.npz").exists(),
            "hybrid_config": (ARTIFACTS / "hybrid_config.json").exists(),
            "siglip_text_cache": (ARTIFACTS / "siglip_text_emb.npz").exists(),
            "evaluation_reports": all(reports.values()),
        },
        "reports_detail": reports,
        "missing_prerequisites": missing,
        "policy_loaded": POLICY_PATH.exists(),
        "instructions": (
            "System is operational and ready for inference."
            if is_ready
            else f"Missing {len(missing)} runtime artifact(s): {missing}. Please transfer these files to artifacts/. Run 'python preflight.py' for details."
        ),
    }


@app.get("/api/meta")
def meta():
    e = get_configured_engine()
    demos = [{"file": p.name, "name": p.stem.split("_", 1)[-1].replace("_", " ")} for p in sorted(DEMO.glob("*.jpg"))] if DEMO.exists() else []

    class_names = e.class_names if e else list(CLASS_INFO.keys())
    cfg = e.cfg if e else {"w_zs": 0.5, "w_lp": 0.1, "w_cn": 0.4, "temperature": 2.0}
    device_name = e.device if e else get_device()

    return {
        "categories": [
            {
                "name": c,
                **CATEGORIES[c],
                "classes": [CLASS_INFO[k]["display"] for k in class_names if k in CLASS_INFO and CLASS_INFO[k]["category"] == c],
            }
            for c in CATEGORY_NAMES
        ],
        "config": cfg,
        "demos": demos,
        "convnext_report": load_json(ARTIFACTS / "evaluation_report.json"),
        "hybrid_report": load_json(ARTIFACTS / "hybrid_report.json"),
        "split": load_json(ARTIFACTS / "split_summary.json"),
        "training_log": load_json(ARTIFACTS / "training_log.json"),
        "class_display": {k: v["display"] for k, v in CLASS_INFO.items()},
        "device": device_name,
        "demonstration_notice": DEMONSTRATION_NOTICE,
        "uncertain_advice": UNCERTAIN_ADVICE,
        "policy_loaded": POLICY_PATH.exists(),
    }


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    gamma: float = Form(1.0),
    glare: float = Form(0.0),
    occlusion: float = Form(0.0),
    blur: int = Form(0),
    conf_threshold: Optional[float] = Form(None),
    entropy_threshold: Optional[float] = Form(None),
    ood_threshold: Optional[float] = Form(None),
):
    data = await image.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image file too large (max 15 MB)")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "The uploaded file is not a readable or supported image format (JPEG, PNG, WebP supported)")
    except Exception:
        raise HTTPException(400, "Could not decode uploaded image")
    img.thumbnail((1024, 1024))

    # Verify engine availability
    eng = engine()

    gamma = min(max(gamma, 0.2), 3.0)
    glare = min(max(glare, 0.0), 1.0)
    occlusion = min(max(occlusion, 0.0), 80.0)
    blur = int(min(max(blur, 0), 12))
    perturbed = abs(gamma - 1) > 1e-3 or glare > 0 or occlusion > 0 or blur > 0
    if perturbed:
        img = apply_perturbations(img, gamma=gamma, glare=glare, occlusion_pct=occlusion, blur=blur)

    overrides = {
        k: v
        for k, v in {
            "conf_threshold": conf_threshold,
            "entropy_threshold": entropy_threshold,
            "ood_threshold": ood_threshold,
        }.items()
        if v is not None
    }

    try:
        with _lock:
            res = eng.predict(img, overrides)
    except Exception as exc:
        raise HTTPException(500, f"Inference execution failed: {type(exc).__name__}")

    status = res.get("status", "ok")
    info = CLASS_INFO.get(res["fine_class"], {})

    if status == "ok":
        res.update(
            {
                "actionable": True,
                "accepted": True,
                "bin": CATEGORIES[res["category"]]["bin"],
                "color": CATEGORIES[res["category"]]["color"],
                "target": info.get("target", CATEGORIES[res["category"]]["bin"]),
                "fine_display": info.get("display", res["fine_class"]),
                "instruction": info.get("instruction", ""),
                "diagnostic_guess": None,
                "guidance": None,
            }
        )
    elif status == "uncertain":
        # Withhold actionable disposal recommendations to avoid recycling contamination
        res.update(
            {
                "actionable": False,
                "accepted": False,
                "bin": None,
                "target": "Action withheld - unconfirmed classification",
                "color": "#d99a06",
                "fine_display": info.get("display", res["fine_class"]),
                "instruction": UNCERTAIN_ADVICE,
                "diagnostic_guess": {
                    "category": res["category"],
                    "fine_class": res["fine_class"],
                    "fine_display": info.get("display", res["fine_class"]),
                    "item": res["item"],
                    "confidence": res["confidence"],
                    "entropy": res["entropy"],
                    "accepted": False,
                    "disclaimer": "Diagnostic hypothesis only. Not accepted for sorting.",
                },
                "guidance": {
                    "action": "withhold_disposal",
                    "reason": "Classification confidence is below the threshold or entropy is elevated.",
                    "steps": [
                        "Check packaging label for recycling symbols (PET 1, HDPE 2, PAP 20-22, ALU 41).",
                        "Retake photo: ensure one item is centered, unobstructed, and well-lit.",
                        "Consult cafeteria staff or signage before discarding unverified items.",
                    ],
                },
            }
        )
    else:  # out_of_scope
        # Withhold actionable disposal recommendations for non-waste
        res.update(
            {
                "actionable": False,
                "accepted": False,
                "bin": None,
                "target": "Action withheld - non-waste item detected",
                "color": "#b4472b",
                "fine_display": info.get("display", res["fine_class"]),
                "instruction": "Action withheld: Image does not appear to be cafeteria waste. Please point the camera at a single waste item or ask staff.",
                "diagnostic_guess": {
                    "category": res["category"],
                    "fine_class": res["fine_class"],
                    "item": res["item"],
                    "non_waste_guess": res.get("non_waste_guess"),
                    "accepted": False,
                    "disclaimer": "Diagnostic hypothesis only. Image flagged as out-of-scope non-waste.",
                },
                "guidance": {
                    "action": "withhold_disposal",
                    "reason": "Image matches non-waste concepts (people, personal electronics, furniture, or background scenery).",
                    "steps": [
                        "Point camera at a single cafeteria waste item.",
                        "Ensure personal belongings or background objects are excluded.",
                        "Consult cafeteria staff if unsure of item categorization.",
                    ],
                },
            }
        )

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
    raise HTTPException(404, "Media file not found")


@app.get("/gallery/{kind}/{name}")
def gallery(kind: str, name: str):
    if kind not in ("failures", "successes"):
        raise HTTPException(404)
    base = (ARTIFACTS / kind).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.exists() or path.suffix.lower() != ".jpg":
        raise HTTPException(404, "Gallery image not found")
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if DEMO.exists():
    app.mount("/demo", StaticFiles(directory=DEMO), name="demo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Binzy Smart Waste Segregation Assistant")
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces (phone on same Wi-Fi)")
    ap.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    ap.add_argument("--device", type=str, default=None, choices=["auto", "mps", "cpu", "cuda"], help="device selection")
    args = ap.parse_args()

    if args.device and args.device != "auto":
        os.environ["BINZY_DEVICE"] = args.device

    uvicorn.run(app, host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port, log_level="info")
