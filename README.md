# Speed Limit Finder: Intelligent Speed Limit Suggestion for Northern Road Conditions

## Overview

This project aims to intelligently suggest appropriate speed limits for roads in northern climates, taking into account real-time weather data, road conditions, and camera-based scene analysis. The system leverages machine learning, data pipelines, and a web-based demo to provide dynamic, context-aware speed limit recommendations, improving road safety and adapting to challenging environments such as snow, ice, and low visibility.

## Project Structure

- **camera_model/**
  - Contains code and resources for training and exporting deep learning models (e.g., ResNet18) to analyze road scenes from camera images. Includes:
    - `train_rscd.py`: Training script for the road scene classification model.
    - `pt_to_onnx.py`: Converts PyTorch models to ONNX format for deployment.
    - `notebooks/`: Jupyter notebooks for model exploration and export.
    - `runs/`: Stores trained model checkpoints and metrics.

- **pipeline/**
  - Houses the main data pipeline and integration logic for the system. Key submodules:
    - `data_pipeline.py`: Orchestrates the flow of data from various sources.
    - `models/`: Contains trained model files (PyTorch `.pt` and ONNX `.onnx`).
    - `speed_limit/`: Logic for speed limit calculation and simulation, including:
      - `nvdb_speed.py`: Main module for NVDB (Norwegian Road Database) speed logic.
      - `nvdb_test_suite.py`: Test suite for speed limit logic.
      - `simulator.py`: Simulates road scenarios.
      - `speed_controller.py`, `speed_features.py`: Feature extraction and control logic.
    - `weather/`: Weather data integration (e.g., `weather_met.py`).
    - `example_videos/`: Example input data for testing.

- **web_demo/**
  - A web-based demonstration app for interacting with the system:
    - `app.py`: Flask (or similar) backend serving the demo.
    - `requirements.txt`: Python dependencies for the web demo.
    - `static/`: Frontend assets (JavaScript, CSS).
    - `templates/`: HTML templates for the web interface.

- **speedlimitFinder.py**
  - Entry point or utility script for running the system or experiments.

## How It Works

1. **Camera Model**: A deep learning model analyzes images/video from road-facing cameras to classify road conditions (e.g., snow, ice, clear).
2. **Weather Data**: Real-time weather information is fetched and processed to assess environmental hazards.
3. **Speed Logic**: The system combines camera and weather inputs with road data (from NVDB) to suggest a safe, context-aware speed limit.
4. **Web Demo**: Users can interact with the system, upload images or videos, and receive speed limit recommendations via a user-friendly web interface.

## Getting Started

1. **Install Dependencies**
   - For the web demo: `pip install -r web_demo/requirements.txt`
   - For model training: See `camera_model/` for additional requirements (e.g., PyTorch, ONNX).

2. **Run the Web Demo**
   - Navigate to `web_demo/` and run `python app.py`.
   - Open your browser to the provided local address.

3. **Train or Export Models**
   - Use scripts in `camera_model/` to train or convert models as needed.

4. **Test and Simulate**
   - Use the test suite and simulator in `pipeline/speed_limit/` to validate logic and run scenario simulations.

## Use Cases

- Dynamic speed limit recommendations for road authorities.
- Enhanced driver assistance systems for northern climates.
- Research and experimentation with multi-modal data fusion for road safety.

## Authors & License

- Developed by NTNU EiT Group 6 (TDT4861)
- For academic and research use. See LICENSE (if available) for details.

---

_This project integrates machine learning, weather data, and road information to make roads safer in challenging northern conditions._
