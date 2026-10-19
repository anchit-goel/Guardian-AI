# Getting Started: Quickstart

> In this page: Zero to first detection in minutes — start the server and run your first image and audio deepfake tests.  
> Who it's for: First-time users, developers evaluating the platform, and quick testers.

---

## 1. Launch the Server

Ensure your virtual environment is active and dependencies are installed:

```bash
# Clone and enter directory
cd /path/to/GuardianAI

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start the unified gateway
python app.py
```

The server binds to `http://0.0.0.0:8080/`. You will see startup diagnostics:
```text
Config path: .../model_runtime_config.json
Selected model: .../models/visual_model.h5
Demo model (untrained placeholder): False
Preprocess mode: rescale_255
Threshold: 0.01
Positive label: REAL
Negative label: FAKE
 * Running on http://0.0.0.0:8080
```

---

## 2. Test 1: Verify Model Status

In a new terminal window, confirm that visual, audio, and multimodal pipelines are loaded:

```bash
curl -s http://127.0.0.1:8080/model-status | python3 -m json.tool
```

Expected output:
```json
{
  "model_loaded": true,
  "audio_model_loaded": true,
  "threshold": 0.01,
  "positive_label": "REAL",
  "negative_label": "FAKE"
}
```

---

## 3. Test 2: Detect an Image

Run inference on any portrait image or the included test image:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -F "image=@modules/multimodal_injection/test_image.jpg" | python3 -m json.tool
```

Expected output:
```json
{
  "label": "REAL",
  "confidence": 99.12,
  "raw_score": 0.9912,
  "processing_time_sec": 0.064,
  "message": "This image is REAL with 99.12% confidence"
}
```

---

## 4. Test 3: Detect Audio Spoofing

Evaluate one of the included genuine or spoofed audio samples:

```bash
curl -s -X POST http://127.0.0.1:8080/predict-audio \
  -F "audio=@modules/audio_spoof_system/wav samples/bonafide1.wav" | python3 -m json.tool
```

Expected output:
```json
{
  "verdict": "GENUINE",
  "risk_score": 0.0435,
  "risk_tier": "LOW",
  "primary_score": 0.0421,
  "models_used": "CNN + LCNN (primary) | Calibrator (standard)",
  "processing_time_sec": 0.038
}
```

---

## 5. Test 4: Run Multimodal Prompt Injection Detection

Evaluate an image for embedded prompt injection:

```bash
curl -s -X POST http://127.0.0.1:8080/predict-multimodal \
  -F "image=@modules/multimodal_injection/test_injection_image.jpg" | python3 -m json.tool
```

Expected output:
```json
{
  "overall_verdict": "SUSPICIOUS — review required",
  "overall_risk_score": 0.455,
  "risk_tier": "MEDIUM",
  "flags": [
    "Prompt injection keywords found: ['ignore']"
  ],
  "signal_breakdown": {
    "ocr_injection": {
      "raw_score": 0.65,
      "contribution": 0.2275
    }
  }
}
```

---

## 6. Open the Web User Interface

Open your browser and navigate to:
```text
http://127.0.0.1:8080/
```
You can drag and drop media files directly into the Audio, Image, or Multimodal tabs and view real-time forensic reports and confidence badges.
