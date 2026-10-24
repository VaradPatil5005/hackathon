#!/usr/bin/env python3
"""
preflight.py - Lightweight system and artifact prerequisite validator for Binzy.

Checks environment, PyTorch device support (MPS / CUDA / CPU), and catalogs
mandatory, optional, and evaluation artifacts without loading full models.

Usage:
    python preflight.py          # Human-readable checklist
    python preflight.py --json   # JSON formatted report
    python preflight.py --quick  # Fast check for startup scripts
"""
import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEMO = ROOT / "demo_samples"
POLICY = ROOT / "policy.json"

ARTIFACT_SPECS = {
    "mandatory": [
        {
            "rel_path": "artifacts/model.pt",
            "size_estimate": "~111 MB",
            "purpose": "Fine-tuned ConvNeXt-Tiny weights checkpoint (12 fine-grained classes)",
            "required_for": "ConvNeXt model inference and hybrid classification",
        },
        {
            "rel_path": "artifacts/siglip_probe.npz",
            "size_estimate": "~56 KB",
            "purpose": "Trained logistic-regression linear probe weights (coef + intercept on SigLIP embeddings)",
            "required_for": "SigLIP linear head expert in hybrid fusion",
        },
        {
            "rel_path": "artifacts/hybrid_config.json",
            "size_estimate": "~350 B",
            "purpose": "Calibrated fusion weights (w_zs, w_lp, w_cn) and safety gate thresholds",
            "required_for": "3-expert probability fusion and out-of-scope/uncertainty gate",
        },
    ],
    "optional_cache": [
        {
            "rel_path": "artifacts/siglip_text_emb.npz",
            "size_estimate": "~1.5 MB",
            "purpose": "Cached text embeddings for 202 waste items + 52 non-waste concepts",
            "required_for": "Fast zero-shot startup (auto-generated if missing)",
        },
        {
            "rel_path": ".cache/hf",
            "size_estimate": "~3.5 GB",
            "purpose": "Local Hugging Face cache for google/siglip-so400m-patch14-384",
            "required_for": "Offline VLM execution (auto-downloaded on first run if online)",
        },
    ],
    "evaluation_reports": [
        {
            "rel_path": "artifacts/split_summary.json",
            "size_estimate": "~1.5 KB",
            "purpose": "Class counts across 70/15/15 train/val/test splits",
            "required_for": "Insights page split breakdown table",
        },
        {
            "rel_path": "artifacts/evaluation_report.json",
            "size_estimate": "~30 KB",
            "purpose": "ConvNeXt clean test metrics, per-class F1, confusion matrices, robustness sweep",
            "required_for": "Insights page KPIs and ConvNeXt detail tab",
        },
        {
            "rel_path": "artifacts/hybrid_report.json",
            "size_estimate": "~15 KB",
            "purpose": "Hybrid vs ConvNeXt vs zero-shot vs probe benchmark and robustness comparisons",
            "required_for": "Insights page model comparison charts and stress test sweeps",
        },
        {
            "rel_path": "artifacts/training_log.json",
            "size_estimate": "~1 KB",
            "purpose": "5-epoch training loss, validation accuracy, and macro-F1 log",
            "required_for": "Insights page training curves",
        },
        {
            "rel_path": "artifacts/split.csv",
            "size_estimate": "~500 KB",
            "purpose": "Exact per-image split assignments (15,520 images)",
            "required_for": "Auditability and reproducibility of data partitions",
        },
        {
            "rel_path": "artifacts/confusion_matrix_5.png",
            "size_estimate": "~50 KB",
            "purpose": "Heatmap of 5-category test confusion matrix",
            "required_for": "Insights page visual inspection",
        },
        {
            "rel_path": "artifacts/confusion_matrix_12.png",
            "size_estimate": "~80 KB",
            "purpose": "Heatmap of 12-class test confusion matrix",
            "required_for": "Insights page fine-grained visual inspection",
        },
        {
            "rel_path": "artifacts/robustness_curves.png",
            "size_estimate": "~100 KB",
            "purpose": "Sensitivity curves for lighting, occlusion, glare, and blur",
            "required_for": "Documentation and presentation deliverables",
        },
    ],
    "demonstration_assets": [
        {
            "rel_path": "demo_samples",
            "size_estimate": "~450 KB (23 files)",
            "purpose": "Bundled test-split samples, degraded items, and out-of-scope non-waste photos",
            "required_for": "One-click demo testing without file uploads",
        },
        {
            "rel_path": "policy.json",
            "size_estimate": "~4.5 KB",
            "purpose": "Customizable cafeteria disposal policy and instructions",
            "required_for": "Decoupled bin assignment and advice customization",
        },
    ],
}


def check_file(rel_path: str):
    p = ROOT / rel_path
    exists = p.exists()
    size = p.stat().st_size if exists and p.is_file() else None
    return exists, size


def check_python_environment():
    env_info = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "system": platform.system(),
        "is_macos": platform.system() == "Darwin",
        "is_arm64": platform.machine() == "arm64",
    }

    # PyTorch checks
    torch_info = {"installed": False}
    try:
        import torch

        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = torch.cuda.is_available()
        torch_info["mps_built"] = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_built()
        torch_info["mps_available"] = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            torch_info["active_device"] = "cuda"
            torch_info["device_name"] = torch.cuda.get_device_name(0)
        elif torch_info.get("mps_available"):
            torch_info["active_device"] = "mps"
            torch_info["device_name"] = "Apple Silicon GPU (Metal Performance Shaders)"
        else:
            torch_info["active_device"] = "cpu"
            torch_info["device_name"] = "CPU fallback"
    except ImportError:
        pass

    env_info["torch"] = torch_info
    return env_info


def inspect_prerequisites():
    env = check_python_environment()
    results = {
        "environment": env,
        "categories": {},
        "summary": {
            "mandatory_missing": 0,
            "optional_missing": 0,
            "reports_missing": 0,
            "demo_assets_missing": 0,
            "ready_for_inference": False,
        },
    }

    for cat_name, items in ARTIFACT_SPECS.items():
        cat_results = []
        for item in items:
            exists, size = check_file(item["rel_path"])
            cat_results.append(
                {
                    "path": item["rel_path"],
                    "exists": exists,
                    "size_bytes": size,
                    "size_estimate": item["size_estimate"],
                    "purpose": item["purpose"],
                    "required_for": item["required_for"],
                }
            )
            if not exists:
                if cat_name == "mandatory":
                    results["summary"]["mandatory_missing"] += 1
                elif cat_name == "optional_cache":
                    results["summary"]["optional_missing"] += 1
                elif cat_name == "evaluation_reports":
                    results["summary"]["reports_missing"] += 1
                elif cat_name == "demonstration_assets":
                    results["summary"]["demo_assets_missing"] += 1

        results["categories"][cat_name] = cat_results

    results["summary"]["ready_for_inference"] = results["summary"]["mandatory_missing"] == 0
    return results


def print_cli_report(data: dict):
    env = data["environment"]
    torch_info = env.get("torch", {})
    summary = data["summary"]

    print("=" * 72)
    print(" ♻️  BINZY SYSTEM & RUNTIME ARTIFACT PREFLIGHT CHECK")
    print("=" * 72)
    print(f" OS / Arch:     {env['platform']} ({env['architecture']})")
    print(f" Python:        {env['python_version']} ({env['python_executable']})")
    if torch_info.get("installed"):
        dev_desc = torch_info.get("device_name", "Unknown")
        print(f" PyTorch:       {torch_info['version']} -> Active Device: {torch_info.get('active_device', 'none').upper()} ({dev_desc})")
        if env["is_macos"]:
            print(f" Apple MPS:     Built={torch_info.get('mps_built')}, Available={torch_info.get('mps_available')}")
    else:
        print(" PyTorch:       NOT INSTALLED (run .venv/bin/pip install -r requirements.txt)")
    print("-" * 72)

    def print_section(title: str, items: list, is_critical: bool = False):
        print(f"\n{title}:")
        for item in items:
            mark = "✅ [PRESENT]" if item["exists"] else ("❌ [MISSING]" if is_critical else "⚠️  [ABSENT] ")
            size_str = f"({item['size_bytes'] / 1024 / 1024:.2f} MB)" if item["size_bytes"] else f"(Expected: {item['size_estimate']})"
            print(f"  {mark} {item['path']} {size_str}")
            print(f"             Purpose: {item['purpose']}")

    print_section("1. Mandatory Runtime Checkpoints (Required for Model Inference)", data["categories"]["mandatory"], is_critical=True)
    print_section("2. Optional Acceleration Caches (Downloaded / Computed automatically if missing)", data["categories"]["optional_cache"])
    print_section("3. Evaluation Reports & Insight Dashboards (Required for UI Insights Tab)", data["categories"]["evaluation_reports"])
    print_section("4. Demonstration Assets & Configuration", data["categories"]["demonstration_assets"])

    print("\n" + "=" * 72)
    if summary["ready_for_inference"]:
        print(" 🎉 STATUS: READY FOR INFERENCE")
        print(" All mandatory model checkpoints are present.")
        print(" Run the application with:  ./run_app.sh")
    else:
        print(f" ⚠️  STATUS: BLOCKED ({summary['mandatory_missing']} mandatory runtime file(s) missing)")
        print("\n ACTIONABLE INSTRUCTIONS FOR TEAMMATE:")
        print(" Your teammate must transfer the following file(s) from their training machine")
        print(f" into the destination directory: {ARTIFACTS}/")
        print()
        for item in data["categories"]["mandatory"]:
            if not item["exists"]:
                print(f"  * Copy: {Path(item['path']).name} ({item['size_estimate']})")
                print(f"    -> Destination: {ARTIFACTS / Path(item['path']).name}")
                print(f"    -> Required for: {item['required_for']}")
        print()
        print(" Optional: Transfer evaluation reports (artifacts/*.json, *.png) for full Insights page metrics.")
        print(" Note: The server will still launch in diagnostic mode so UI inspection remains possible.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Binzy Runtime Preflight Validator")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--quick", action="store_true", help="Fast exit code without printing full report")
    args = parser.parse_args()

    results = inspect_prerequisites()

    if args.json:
        print(json.dumps(results, indent=2))
    elif not args.quick:
        print_cli_report(results)

    sys.exit(0 if results["summary"]["ready_for_inference"] else 1)


if __name__ == "__main__":
    main()
