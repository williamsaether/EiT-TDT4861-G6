# Speed Limit Finder: Intelligent Speed Limit Suggestion for Northern Road Conditions

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

# 3. Install Python dependencies
pip install -r web_demo/requirements.txt

# 4. Run the web demo
python web_demo/app.py

# 5. Open http://localhost:8000 in your browser
```

Or use the convenience script:

```bash
chmod +x run.sh && ./run.sh
```

## Project Structure

- **web_demo/**
  - `app.py`: Flask backend — runs ONNX inference on the demo videos and serves the UI.
  - `requirements.txt`: Python dependencies for the web demo.
  - `static/`: Frontend assets (JavaScript, CSS).
  - `templates/`: HTML templates.

- **pipeline/**
  - `models/`: ONNX model files used for road-surface classification at runtime.
  - `speed_limit/`: Speed limit calculation and simulation logic.
  - `weather/`: Weather data integration.
  - `data_pipeline.py`: Orchestrates the data flow from various sources.

- **camera_model/**
  - `train_rscd.py`: Training script for the road scene classification model (ResNet18).
  - `pt_to_onnx.py`: Converts a trained PyTorch checkpoint to ONNX format.
  - `notebooks/`: Jupyter notebooks for model exploration and export.

- **videos/**
  - 10 demo MP4 clips (stored in Git LFS) with matching `metadata.json` containing GPS, weather, and speed-limit ground-truth data.

- **merged_survey.csv**: Human survey data with recommended speeds per video, used to evaluate model accuracy.

## How It Works

1. **Camera Model**: A ResNet18 model classifies each video frame into road-surface categories (dry asphalt, wet asphalt, fresh snow, ice, etc.).
2. **Weather Data**: Temperature and precipitation from `metadata.json` produce a weather reduction factor.
3. **Speed Recommendation**: Camera confidence and weather factor are blended with configurable weights to produce a safe recommended speed.
4. **Web Demo**: The UI lets you scrub through each video, view per-frame predictions, and tune the blending parameters live. An auto-tune feature runs differential evolution to find optimal parameters against human survey targets.

## Training / Exporting a New Model

```bash
# Train
python camera_model/train_rscd.py

# Export the best checkpoint to ONNX
python camera_model/pt_to_onnx.py
# Places the new .onnx file in pipeline/models/
```

The web demo automatically discovers all `.onnx` files in `pipeline/models/` and lets you switch between them in the UI.

## Authors & License

- Developed by NTNU EiT Group 6 (TDT4861)
- For academic and research use. See LICENSE (if available) for details.

---

_This project integrates machine learning, weather data, and road information to make roads safer in challenging northern conditions._
