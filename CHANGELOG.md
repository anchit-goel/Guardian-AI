# Changelog

All notable changes to the Guardian AI codebase and forensic pipelines will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Visual Forensic Pipeline (`POST /predict`)**:
  - ResNet50 convolutional neural network with custom multi-layer classification head (`models/visual_model.h5`, `models/new_model.h5`).
  - Robust Keras/HDF5 deserializer (`load_model_robust`) stripping legacy `DepthwiseConv2D` attributes.
  - Dynamic thresholding via `model_runtime_config.json` with calibrated sensitivity default ($\tau = 0.01$).
  - Offline threshold calibration utility (`scripts/calibrate_threshold.py`) optimizing for validation F1-score.
  - High-throughput local directory batch scoring script (`scripts/generate_scores_local.py`).
- **Acoustic Anti-Spoofing Pipeline (`POST /predict-audio`)**:
  - Dual-feature audio representations: 80-bin Log-Mel Spectrogram and 60-coefficient Linear Frequency Cepstral Coefficients (LFCC).
  - 3-stage Convolutional Network with Squeeze-and-Excitation channel attention (`cnn_best_clean.pth`).
  - 8-layer Light Convolutional Neural Network (LCNN) with Max-Feature-Map (MFM) non-linearities (`lcnn_best_clean.pth`).
  - Weighted primary ensemble ($0.55 \cdot \text{CNN} + 0.45 \cdot \text{LCNN}$).
  - Disagreement-aware secondary calibrator using `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`.
  - Extreme probability softening and 4-tier risk classification (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **Multimodal Prompt Injection Engine (`POST /predict-multimodal`)**:
  - Pixel-level local variance analysis over $32 \times 32$ tiles detecting low-contrast "ghost" text.
  - Optical Character Recognition via EasyOCR with confidence thresholding.
  - Fuzzy sliding-window prompt injection keyword scanner matching against an 18-phrase adversarial catalog.
  - Zero-shot visual captioning via `Salesforce/blip-image-captioning-base`.
  - Speech-to-text audio transcription via `openai/whisper-base`.
  - Cross-modal narrative consistency verification via `openai/clip-vit-base-patch32` and from-scratch TF-IDF/Jaccard semantic similarity.
  - Transparent weighted risk aggregation and signal breakdown reporting.
- **Web Serving & User Interface**:
  - Flask HTTP REST gateway (`app.py`) with CORS middleware and model diagnostic endpoint (`GET /model-status`).
  - Standalone FastAPI microservice (`modules/multimodal_injection/main.py :8001`).
  - React 18 single-page application (`index.html`) with dark-mode design system (`style.css`), drag-and-drop dropzones, and real-time forensic reports.
  - Render cloud deployment blueprint (`render.yaml`) with Python 3.10.13 pin and Gunicorn multi-threading configuration.
- **Explainable AI (XAI) Suite**:
  - Grad-CAM heatmap visualization scripts for convolutional feature localization.
  - Integrated Gradients (IG), Layer-wise Relevance Propagation (LRP), Saliency Maps, SHAP, and LIME interpretability implementations.
- **Documentation Overhaul**:
  - Comprehensive `/docs` directory covering concepts, models, datasets, training guides, verified benchmark evaluations, architecture decision records (ADRs), API references, and migration audit.
