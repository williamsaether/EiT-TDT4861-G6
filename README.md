<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary><h1>F.A.R.T: Føre-Adaptiv Risiko-Teknologi</h1></summary>
    <summary><p>Traction-Adaptive Risk Technology. Intelligent Speed Limit Suggestion for Northern Road Conditions.
</p></summary>
  </ul>
</div>

## Overview

This project aims to intelligently suggest appropriate speed limits for roads in northern climates, taking into account real-time weather data, road conditions, and camera-based scene analysis. The system leverages machine learning, data pipelines, and a web-based demo to provide dynamic, context-aware speed limit recommendations, improving road safety and adapting to challenging environments such as snow, ice, and low visibility.

## Prerequisites

- Python 3.9+
- [Git LFS](https://git-lfs.com/) — required to download video and model files

  ```bash
  # macOS
  brew install git-lfs

  # Ubuntu / Debian
  sudo apt install git-lfs

  # Windows (with winget)
  winget install GitHub.GitLFS
  ```

## Quickstart

```bash
# 1. Clone the repo (Git LFS fetches videos and models automatically)
git clone https://github.com/<your-org>/<repo>.git
cd <repo>

# 2. If you cloned before installing Git LFS, fetch the large files manually
git lfs pull

# 3. Run the unified app (driving demo + tuning dashboard)
./run.sh

# 4. Open one of:
#    http://localhost:8000/driving/
#    http://localhost:8000/driving/examples
#    http://localhost:8000/tuning/
```

## Project Structure

- **app/**
  - `app.py`: Single app entrypoint.
  - `__init__.py`: Flask app factory and blueprint registration.
  - `common/`: Shared logic/constants (tuned parameters + recommendation helpers).
    - `common/providers/`: Shared provider core for NVDB speed limits and weather lookup.
    - `common/core/context_provider.py`: Shared context pipeline used by web and reusable scripts.
  - `driving/`: Driving demo routes, services, templates, static assets.
  - `tuning/`: Auto-tuning dashboard split into focused modules:
    - `routes.py`: API layer only.
    - `config.py`: constants + metadata/video loading.
    - `state.py`: tunable state model.
    - `analyzer.py`: ONNX frame analysis pipeline.
    - `recommendation_service.py`: recommendation aggregation logic.
    - `autotune_service.py`: differential evolution optimizer.
    - `templates/` + `static/`: UI assets.
  - `requirements.txt`: Dependencies for both driving and tuning endpoints.

- **research/**
  - **Purpose**: experimentation, testing, and feature exploration (not the primary runtime path for the web app).
  - `speed_limit/`: exploratory speed-limit logic, simulations, and test helpers.
  - `weather/`: exploratory weather integration variants.
  - `data_pipeline.py`: research-style orchestration for trying ideas and provider combinations.
  - Production endpoints under `/driving` and `/tuning` use code in `app/` as the source of truth.

- **models/**
  - `checkpoints/`: PyTorch checkpoints (`.pt`) used for training/evaluation.
  - `onnx/`: Runtime ONNX models discovered by tuning and research tools.
  - `eval/`: Evaluation summaries (`eval_summary*.json`).
  - `manifest.json`: Canonical inventory of model artifacts.

- **camera_model/**
  - `train_rscd.py`: Training script for the road scene classification model (ResNet18).
  - `pt_to_onnx.py`: Converts a trained PyTorch checkpoint to ONNX format.
  - `notebooks/`: Jupyter notebooks for model exploration and export.

- **videos/**
  - 10 demo MP4 clips (stored in Git LFS) with matching `metadata.json` containing GPS, weather, and speed-limit ground-truth data.

- **merged_survey.csv**: Human survey data with recommended speeds per video, used to evaluate model accuracy.

## How It Works

1. **Driving Demo (`/driving`)**:
   - Frontend captures camera frames and performs local classification in-browser (simulating on-device/car inference).
   - Backend receives classification + confidence, fetches weather context, resolves speed limit from Norwegian NVDB (`nvdbapiles.atlas.vegvesen.no`) with metadata fallback, and computes recommended speed.
   - UI overlays current speed limit, weather, classification, and recommended speed on top of the live camera feed.
   - Example-video mode (`/driving/examples`) uses only `metadata.json` entries with `forTraining: false`.
2. **Auto-Tuning Dashboard (`/tuning`)**:
   - Runs ONNX inference on the prerecorded videos.
   - Lets you tune camera/weather blending parameters and compare against survey targets.

## Training / Exporting a New Model

```bash
# Train
python camera_model/train_rscd.py
# (saves checkpoints under runs/rscd_resnet18 by default)

# Export the best checkpoint to ONNX
python camera_model/pt_to_onnx.py --pt models/checkpoints/rscd_resnet18_v2.pt --onnx models/onnx/rscd_resnet18_v2.onnx

# Evaluate checkpoints
python camera_model/evaluate_saved_models.py
# (reads models/checkpoints and writes models/eval/eval_summary.json by default)
```

The web demo automatically discovers all `.onnx` files in `models/onnx/` and lets you switch between them in the UI.

## Authors & License

- Developed by NTNU EiT Group 6 (TDT4861)
- For academic and research use. See LICENSE (if available) for details.

---

_This project integrates machine learning, weather data, and road information to make roads safer in challenging northern conditions._
