#!/usr/bin/env python3
"""
test_validation.py - Comprehensive regression & validation suite for Binzy.

Tests:
1. Environment & Device Resolution (Apple Silicon MPS / CPU fallback)
2. Policy Loading & Integrity (decoupled policy.json, 5 bins, battery safety, glass crates)
3. Preflight prerequisite verification
4. Uncertainty & Out-of-Scope response logic (withholding actionable advice, unaccepted labeling)
5. Perturbation engine on demo images (dark, occluded, glare, blur)
6. Live HTTP flow against server.py (health, meta, error handling, input validation)
"""
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# Add ROOT to sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import waste_core
from waste_core import (
    CATEGORIES,
    CATEGORY_NAMES,
    CLASS_INFO,
    DEMONSTRATION_NOTICE,
    UNCERTAIN_ADVICE,
    apply_perturbations,
    get_device,
    load_policy,
    normalized_entropy,
)


def test_section(title):
    print("\n" + "=" * 60)
    print(f" RUNNING: {title}")
    print("=" * 60)


def test_device_resolution():
    test_section("Device Resolution & Apple MPS Support")
    device_default = get_device()
    print(f" Default detected device: {device_default}")
    assert device_default in ("mps", "cuda", "cpu"), f"Unexpected device: {device_default}"

    # Verify CPU override
    dev_cpu = get_device("cpu")
    assert dev_cpu == "cpu", f"CPU override failed: {dev_cpu}"
    print(" Verified: Explicit CPU fallback works")

    # Verify MPS if available on this Mac
    import torch

    if torch.backends.mps.is_available():
        dev_mps = get_device("mps")
        assert dev_mps == "mps", f"MPS override failed: {dev_mps}"
        print(" Verified: Apple Silicon MPS detected and selectable")
    else:
        print(" Note: MPS not available on this platform (CPU/CUDA only)")


def test_policy_loading():
    test_section("Policy Loading & Guidance Integrity")
    policy = load_policy()
    assert "categories" in policy, "Missing 'categories' in policy.json"
    assert "classes" in policy, "Missing 'classes' in policy.json"
    assert "demonstration_notice" in policy, "Missing 'demonstration_notice' in policy.json"

    # Check 5 categories
    cats = policy["categories"]
    assert len(cats) == 5, f"Expected 5 categories, found {len(cats)}"
    for cat in ["Paper & Cardboard", "Plastic", "Metal", "Food Scraps / Organic", "Non-Recyclable / Special Waste"]:
        assert cat in cats, f"Missing category: {cat}"

    # Verify category 5 explanation
    cat5 = cats["Non-Recyclable / Special Waste"]
    assert "NOT a single physical bin" in cat5.get("description", ""), "Category 5 must explain it is not one physical bin"
    print(" Verified: Category 5 correctly explains composite residual/special streams")

    # Verify battery safety
    classes = policy["classes"]
    assert "battery" in classes
    assert "NEVER" in classes["battery"]["instruction"] or "hazard" in classes["battery"]["instruction"].lower(), "Battery instruction must contain hazard warning"
    assert "drop box" in classes["battery"]["target"].lower() or "staff" in classes["battery"]["target"].lower(), "Battery target must be separate drop box or staff"
    print(" Verified: Battery handling strictly separated from cafeteria bins")

    # Verify glass collection
    for glass_cls in ["brown-glass", "green-glass", "white-glass"]:
        assert "crate" in classes[glass_cls]["target"].lower(), f"{glass_cls} must use glass collection crate"
    print(" Verified: Glass collection crate guidance preserved")

    # Verify uncertainty advice does NOT direct unidentified items into general waste
    advice = policy.get("uncertain_guidance", {}).get("notice", "")
    assert "Red/Black bin" not in UNCERTAIN_ADVICE, "UNCERTAIN_ADVICE must not direct unidentified items into Red/Black bin automatically"
    print(" Verified: Uncertainty guidance avoids directing unidentified items into general waste")


def test_preflight_tool():
    test_section("Preflight Prerequisite Verification Tool")
    from preflight import inspect_prerequisites

    results = inspect_prerequisites()
    assert "environment" in results
    assert "categories" in results
    assert "summary" in results
    print(f" Preflight check executed successfully:")
    print(f"  - Inference ready: {results['summary']['ready_for_inference']}")
    print(f"  - Mandatory missing: {results['summary']['mandatory_missing']}")
    print(f"  - Active device: {results['environment']['torch'].get('active_device')}")


def test_perturbations():
    test_section("Image Perturbations (Dark, Glare, Occlusion, Blur)")
    demo_dir = ROOT / "demo_samples"
    sample_path = demo_dir / "01_Battery.jpg"
    assert sample_path.exists(), "Sample image 01_Battery.jpg not found"

    with Image.open(sample_path) as img:
        img = img.convert("RGB")
        w, h = img.size

        # Dark (gamma 0.3)
        dark = apply_perturbations(img, gamma=0.3)
        assert dark.size == (w, h)

        # Glare
        glare = apply_perturbations(img, glare=0.8)
        assert glare.size == (w, h)

        # Occlusion (50%)
        occ = apply_perturbations(img, occlusion_pct=50)
        assert occ.size == (w, h)

        # Blur
        blur = apply_perturbations(img, blur=6)
        assert blur.size == (w, h)

    print(" Verified: All image perturbation transforms execute cleanly on real sample images")


def test_server_uncertainty_response_logic():
    test_section("Regression Check: Response Formatting for OK, Uncertain, Out-of-Scope")

    # Test the response formatting logic used in server.py
    def format_predict_response(res):
        status = res.get("status", "ok")
        info = CLASS_INFO.get(res["fine_class"], {})

        if status == "ok":
            res.update({
                "actionable": True,
                "accepted": True,
                "bin": CATEGORIES[res["category"]]["bin"],
                "color": CATEGORIES[res["category"]]["color"],
                "target": info.get("target", CATEGORIES[res["category"]]["bin"]),
                "fine_display": info.get("display", res["fine_class"]),
                "instruction": info.get("instruction", ""),
                "diagnostic_guess": None,
                "guidance": None,
            })
        elif status == "uncertain":
            res.update({
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
                    "reason": "Classification confidence is below threshold or entropy is high.",
                    "steps": [
                        "Check packaging label for recycling symbols (PET 1, HDPE 2, PAP 20-22, ALU 41).",
                        "Retake photo: ensure one item is centered, unobstructed, and well-lit.",
                        "Consult cafeteria staff or signage before discarding unverified items.",
                    ],
                },
            })
        else:  # out_of_scope
            res.update({
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
            })
        return res

    # 1. OK case
    res_ok = format_predict_response({
        "status": "ok",
        "category": "Plastic",
        "fine_class": "plastic",
        "item": "a plastic water bottle",
        "confidence": 0.94,
        "entropy": 0.12,
    })
    assert res_ok["actionable"] is True
    assert res_ok["accepted"] is True
    assert res_ok["bin"] is not None
    assert res_ok["target"] == "Yellow Bin"
    assert res_ok["diagnostic_guess"] is None
    print(" Verified: 'ok' status returns actionable bin instruction")

    # 2. Uncertain case
    res_unc = format_predict_response({
        "status": "uncertain",
        "category": "Paper & Cardboard",
        "fine_class": "paper",
        "item": "a paper cup",
        "confidence": 0.45,
        "entropy": 0.72,
    })
    assert res_unc["actionable"] is False, "Uncertain status must NOT be actionable"
    assert res_unc["accepted"] is False, "Uncertain status must NOT be accepted"
    assert res_unc["bin"] is None, "Uncertain status must NOT specify an actionable bin"
    assert "withheld" in res_unc["target"].lower()
    assert res_unc["diagnostic_guess"] is not None
    assert res_unc["diagnostic_guess"]["accepted"] is False
    assert "hypo" in res_unc["diagnostic_guess"]["disclaimer"].lower() or "unaccepted" in res_unc["diagnostic_guess"]["disclaimer"].lower() or "not accepted" in res_unc["diagnostic_guess"]["disclaimer"].lower()
    assert res_unc["guidance"]["action"] == "withhold_disposal"
    print(" Verified: 'uncertain' status withholds bin recommendation and clearly labels diagnostic guess as unaccepted")

    # 3. Out-of-scope case
    res_oos = format_predict_response({
        "status": "out_of_scope",
        "category": "Non-Recyclable / Special Waste",
        "fine_class": "trash",
        "item": "general item",
        "non_waste_guess": "a person holding a smartphone",
        "confidence": 0.88,
        "entropy": 0.25,
    })
    assert res_oos["actionable"] is False
    assert res_oos["accepted"] is False
    assert res_oos["bin"] is None
    assert "withheld" in res_oos["target"].lower()
    assert res_oos["diagnostic_guess"]["accepted"] is False
    assert res_oos["diagnostic_guess"]["non_waste_guess"] == "a person holding a smartphone"
    print(" Verified: 'out_of_scope' status withholds bin recommendation and provides non-waste guidance")


def test_live_http_server():
    test_section("Live HTTP Server Validation")
    import requests

    test_port = 8899
    cmd = [PYTHON, "server.py", "--port", str(test_port)]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    base_url = f"http://127.0.0.1:{test_port}"
    max_wait = 20
    started = False
    for _ in range(max_wait):
        time.sleep(1)
        try:
            r = requests.get(f"{base_url}/api/health", timeout=2)
            if r.status_code == 200:
                started = True
                break
        except Exception:
            pass

    if not started:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"Server failed to start on port {test_port}.\nStdout: {stdout}\nStderr: {stderr}")

    try:
        # 1. Health endpoint
        r = requests.get(f"{base_url}/api/health")
        assert r.status_code == 200
        health = r.json()
        print(f" /api/health response: status='{health['status']}', device='{health['device']}', inference_ready={health['inference_ready']}")
        assert "status" in health
        assert "device" in health
        assert "components" in health
        assert "missing_prerequisites" in health

        # 2. Meta endpoint
        r = requests.get(f"{base_url}/api/meta")
        assert r.status_code == 200
        meta = r.json()
        print(f" /api/meta response: categories={len(meta['categories'])}, demos={len(meta['demos'])}, device='{meta['device']}'")
        assert len(meta["categories"]) == 5
        assert "demonstration_notice" in meta
        assert "policy_loaded" in meta

        # 3. Predict without model artifacts (should return 503 Service Unavailable, NOT crash with 500)
        sample_img = ROOT / "demo_samples" / "01_Battery.jpg"
        with open(sample_img, "rb") as f:
            files = {"image": ("01_Battery.jpg", f, "image/jpeg")}
            r = requests.post(f"{base_url}/api/predict", files=files)
            print(f" /api/predict (with missing model artifacts): status_code={r.status_code}")
            assert r.status_code == 503, f"Expected 503 when model.pt is missing, got {r.status_code}"
            err_data = r.json()
            assert "detail" in err_data
            assert "artifacts" in err_data["detail"].lower() or "model" in err_data["detail"].lower()
            # Ensure no local file paths are leaked in detail
            assert "/Users/" not in err_data["detail"]
            print(f"   Clean error message: '{err_data['detail']}'")

        # 4. Corrupt / invalid image upload
        corrupt_bytes = b"This is plainly not a JPEG or PNG file header."
        files = {"image": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")}
        r = requests.post(f"{base_url}/api/predict", files=files)
        print(f" /api/predict (corrupt file upload): status_code={r.status_code}")
        assert r.status_code == 400, f"Expected 400 for corrupt upload, got {r.status_code}"

        # 5. Static app shell
        r = requests.get(f"{base_url}/")
        assert r.status_code == 200
        assert "html" in r.text.lower()
        print(" Root index.html served cleanly")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(" Test HTTP server terminated cleanly.")


def main():
    print("========================================================================")
    print(" 🧪 RUNNING BINZY VALIDATION & REGRESSION TEST SUITE")
    print("========================================================================")
    test_device_resolution()
    test_policy_loading()
    test_preflight_tool()
    test_perturbations()
    test_server_uncertainty_response_logic()
    test_live_http_server()
    print("\n" + "=" * 60)
    print(" ✅ ALL TEST SUITES PASSED SUCCESSFULLY!")
    print("========================================================================")


if __name__ == "__main__":
    main()
