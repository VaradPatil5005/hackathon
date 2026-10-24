# ♻️ Binzy: Smart Waste Segregation Assistant

VIT AI Innovate-A-Thon, **Question 6**. You point a camera at any cafeteria waste item, and Binzy names the item, tells you which of the **5 bins** it goes in, and explains how to dispose of it. If the photo isn't waste, or Binzy isn't confident, it says so and asks you to check the label or ask staff.

## Run the demo

Double-click **`run_app.bat`**. It opens http://localhost:8000 after the models load (about 30 s).
To scan from a phone on the same Wi-Fi, use `run_app.bat --lan` and open `http://<laptop-ip>:8000`.

The app has four pages:
- **Scan:** camera, upload, drag-and-drop or paste, plus a live auto-scan mode.
- **Insights:** the dataset split, all metrics, the external-dataset comparison and the failure cases.
- **Stress test:** lighting, glare, occlusion and blur sliders.
- **How it works:** the method, Responsible AI notes and SDG alignment.

## How it recognises *any* item

| Expert | What it does |
|---|---|
| SigLIP-so400m zero-shot | Compares the photo with 202 written item descriptions. Examples: "a Kit Kat wrapper", "a half-eaten piece of cake", "a clay kulhad tea cup", "a Frooti tetra pack". It also compares against 52 "not waste" concepts such as people, phones, animals and scenery. |
| SigLIP + trained head | A logistic-regression head on SigLIP image features, trained on the 70% training split. |
| ConvNeXt-Tiny | Fully fine-tuned on the training split, with lighting and occlusion augmentation. |

The three experts' probabilities are fused with weights tuned on validation data and summed into the 5 bins. A safety gate then decides what to show:
- **Out of scope:** the "not waste" probability is too high.
- **Not sure:** the bin confidence is too low or the entropy is too high.

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
| `server.py` | FastAPI backend and the web app |
| `static/` | design system: tokens, Plus Jakarta Sans and Mukta fonts, components |
| `hybrid.py` · `waste_vocab.py` | The hybrid recogniser and its open vocabulary |
| `waste_core.py` | Categories, instructions, ConvNeXt, image perturbations |
| `train.py` · `evaluate.py` | 70/15/15 split, ConvNeXt training and test-set evaluation |
| `build_hybrid.py` | SigLIP features, linear head, fusion tuning, external and robustness evaluation |
| `generate_demo_samples.py` · `build_deliverables.py` | Demo images, PPT, report and charts |
| `artifacts/` | Models, split, all reports |
| `deliverables/` | `Smart_Waste_Segregation_Assistant.pptx`, `REPORT.md`, charts, screenshots |
| `legacy/streamlit_app.py` | The first Streamlit prototype, kept for reference only |

To rebuild everything from scratch, run `run_pipeline.bat`. All data, models and caches live inside this folder.
