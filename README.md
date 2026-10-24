# ♻️ Binzy: Smart Waste Segregation Assistant

VIT AI Innovate-A-Thon, **Question 6**. Point a camera at cafeteria waste, and Binzy recognizes the item, suggests its disposal category across the **5 bins**, and provides handling instructions. When the photo is not waste, or when confidence is low, actionable disposal recommendations are withheld to prevent recycling contamination—prompting the user to check the label, retake the photo, or consult staff.

## Run the demo

### macOS (Apple Silicon MPS or CPU)
1. **Quick setup:**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```
2. **Preflight check:**
   ```bash
   .venv/bin/python preflight.py
   ```
3. To scan from a phone on the same Wi-Fi, use `./run_app.sh --lan` and open `http://<laptop-ip>:8000`.

### Windows
Double-click **`run_app.bat`**. It opens http://localhost:8000 after the models load.
To scan from a phone on the same Wi-Fi, use `run_app.bat --lan` and open `http://<laptop-ip>:8000`.

The app has four pages:
- **Scan:** camera, upload, drag-and-drop or paste, plus a live auto-scan mode.
- **Insights:** the dataset split, all metrics, the external-dataset comparison and failure cases.
- **Stress test:** lighting, glare, occlusion and blur sliders.
- **How it works:** the method, Responsible AI notes, policy customization and SDG alignment.

## How it recognises open-vocabulary items

| Expert | What it does |
|---|---|
| SigLIP-so400m zero-shot | Compares the photo with 202 written item descriptions. Examples: "a Kit Kat wrapper", "a half-eaten piece of cake", "a clay kulhad tea cup", "a Frooti tetra pack". It also compares against 52 "not waste" concepts such as people, phones, animals and scenery. |
| SigLIP + trained head | A logistic-regression head on SigLIP image features, trained on the 70% training split. |
| ConvNeXt-Tiny | Fully fine-tuned on the training split, with lighting and occlusion augmentation. |

The three experts' probabilities are fused with weights tuned on validation data and summed into the 5 bins. A safety gate then decides what to show:
- **Out of scope:** the "not waste" probability is too high (actionable bin instruction withheld).
- **Not sure:** the bin confidence is too low or the entropy is too high (actionable bin instruction withheld; diagnostic hypothesis labeled as unaccepted).

## Results (photos never used for training)

| Dataset | ConvNeXt only | **Hybrid** |
|---|---|---|
| Kaggle test split (2,328) | 98.8% | **99.5%** |
| TrashNet (2,527, different source; used for tuning) | 93.4% | **95.6%** |
| External set #2 (397, fully held out) | 86.7% | **89.9%** |
| Non-waste photos refused (1,000 Imagenette) | 68.7% | **93.0%** |

`deliverables/REPORT.md` has the full results.

## Project layout

| Path | Purpose |
|---|---|
| `server.py` | FastAPI backend and the web app with /api/health and /api/predict |
| `policy.json` | Editable cafeteria disposal policy and instructions (no retraining needed) |
| `preflight.py` | Lightweight prerequisite and artifact verification CLI |
| `run_app.sh` · `run_app.bat` | macOS (Apple MPS/CPU) and Windows launchers |
| `static/` | Design system: tokens, Plus Jakarta Sans and Mukta fonts, components |
| `hybrid.py` · `waste_vocab.py` | The hybrid recogniser and its open vocabulary |
| `waste_core.py` | Categories, policy loader, device resolution, ConvNeXt, perturbations |
| `train.py` · `evaluate.py` | 70/15/15 split, ConvNeXt training and test-set evaluation |
| `build_hybrid.py` | SigLIP features, linear head, fusion tuning, external and robustness evaluation |
| `generate_demo_samples.py` · `build_deliverables.py` | Demo images, PPT, report and charts |
| `artifacts/` | Models, split, all reports |
| `deliverables/` | `Smart_Waste_Segregation_Assistant.pptx`, `REPORT.md`, charts, screenshots |
| `legacy/streamlit_app.py` | The first Streamlit prototype, kept for reference only |

To rebuild everything from scratch, run `run_pipeline.bat`. All data, models and caches live inside this folder.
