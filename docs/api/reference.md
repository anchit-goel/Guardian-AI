# API: Endpoint Reference

> In this page: Complete endpoint specification for Guardian AI, including query parameters, form-data schemas, status codes, and copy-paste curl and Python examples.  
> Who it's for: Full-stack developers, API consumers, and automation engineers.

---

## Endpoint Catalog

1. [`GET /model-status`](#1-get-model-status) — Model health & diagnostic status
2. [`POST /predict`](#2-post-predict) — Visual deepfake classification
3. [`POST /predict-audio`](#3-post-predict-audio) — Acoustic voice spoof detection
4. [`POST /predict-multimodal`](#4-post-predict-multimodal) — Multimodal & prompt injection analysis
5. [Dedicated FastAPI Microservice Endpoints](#5-dedicated-fastapi-microservice-endpoints)

---

## 1. `GET /model-status`

Returns runtime diagnostics and load states for visual, audio, and multimodal subsystems.

### HTTP Request
`GET /model-status`

### curl Example
```bash
curl -s http://127.0.0.1:8080/model-status
```

### Python Example
```python
import requests

response = requests.get("http://127.0.0.1:8080/model-status")
print(response.json())
```

### Response Schema (`200 OK`)
```json
{
  "config_path": "/opt/render/project/src/model_runtime_config.json",
  "model_path": "/opt/render/project/src/models/visual_model.h5",
  "input_shape": [null, 224, 224, 3],
  "output_shape": [null, 1],
  "preprocess_mode": "rescale_255",
  "threshold": 0.01,
  "positive_label": "REAL",
  "negative_label": "FAKE",
  "input_size": [224, 224],
  "model_loaded": true,
  "model_load_error": null,
  "is_demo_model": false,
  "audio_root": "/opt/render/project/src/modules/audio_spoof_system",
  "audio_model_loaded": true,
  "audio_model_error": null,
  "multimodal_root": "/opt/render/project/src/modules/multimodal_injection",
  "multimodal_model_loaded": true,
  "multimodal_model_loading": false,
  "multimodal_model_error": null
}
```

---

## 2. `POST /predict`

Evaluates a facial image for synthetic manipulation or deepfake generation.

### HTTP Request
`POST /predict`  
`Content-Type: multipart/form-data`

### Form-Data Parameters
| Key | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | Binary File | **Yes** | Input image file (`.jpg`, `.png`, `.webp`, `.bmp`). |

### curl Example
```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "image=@/path/to/portrait.jpg"
```

### Python Example
```python
import requests

url = "http://127.0.0.1:8080/predict"
with open("sample_face.jpg", "rb") as f:
    files = {"image": ("sample_face.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.status_code, response.json())
```

### Response Schema (`200 OK`)
```json
{
  "label": "REAL",
  "confidence": 98.45,
  "raw_score": 0.9845,
  "processing_time_sec": 0.082,
  "message": "This image is REAL with 98.45% confidence",
  "model_path": "/path/to/models/visual_model.h5",
  "preprocess_mode": "rescale_255",
  "threshold": 0.01,
  "positive_label": "REAL",
  "negative_label": "FAKE",
  "is_demo_model": false
}
```

---

## 3. `POST /predict-audio`

Evaluates an audio clip for synthetic speech, TTS voice cloning, or voice conversion artifacts.

### HTTP Request
`POST /predict-audio`  
`Content-Type: multipart/form-data`

### Form-Data Parameters
| Key | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `audio` | Binary File | **Yes** | Speech recording (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`). |

### curl Example
```bash
curl -X POST http://127.0.0.1:8080/predict-audio \
  -F "audio=@/path/to/speech.wav"
```

### Python Example
```python
import requests

url = "http://127.0.0.1:8080/predict-audio"
with open("voice_sample.wav", "rb") as f:
    files = {"audio": ("voice_sample.wav", f, "audio/wav")}
    response = requests.post(url, files=files)

print(response.json())
```

### Response Schema (`200 OK`)
```json
{
  "cnn_score": 0.1245,
  "lcnn_score": 0.1582,
  "calibrator_score": 0.1410,
  "primary_score": 0.1397,
  "calibration_mode": "standard",
  "risk_score": 0.1400,
  "verdict": "GENUINE",
  "risk_tier": "LOW",
  "models_used": "CNN + LCNN (primary) | Calibrator (standard)",
  "explanation": "Primary models — CNN: 12.5%, LCNN: 15.8%. Calibrator: 14.1%. Final risk: 14.0% (LOW)",
  "audio_file": "speech.wav",
  "processing_time_sec": 0.048
}
```

---

## 4. `POST /predict-multimodal`

Performs combined cross-modal consistency analysis, low-contrast ghost text detection, OCR scanning, and prompt injection detection.

### HTTP Request
`POST /predict-multimodal`  
`Content-Type: multipart/form-data`

### Form-Data Parameters
| Key | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `image` | Binary File | Optional* | Visual image file (`.jpg`, `.png`, etc.). |
| `audio` | Binary File | Optional* | Audio speech file (`.wav`, `.mp3`, etc.). |

*\*At least one of `image` or `audio` must be provided.*

### curl Example (Both Modalities)
```bash
curl -X POST http://127.0.0.1:8080/predict-multimodal \
  -F "image=@/path/to/image.jpg" \
  -F "audio=@/path/to/audio.wav"
```

### Python Example
```python
import requests

url = "http://127.0.0.1:8080/predict-multimodal"
with open("slide.png", "rb") as img, open("narration.wav", "rb") as aud:
    files = {
        "image": ("slide.png", img, "image/png"),
        "audio": ("narration.wav", aud, "audio/wav"),
    }
    response = requests.post(url, files=files)

print(response.json())
```

### Response Schema (`200 OK`)
```json
{
  "overall_risk_score": 0.4550,
  "overall_verdict": "SUSPICIOUS — review required",
  "risk_tier": "MEDIUM",
  "flags": [
    "Prompt injection keywords found: ['ignore']"
  ],
  "signal_breakdown": {
    "hidden_text": {
      "raw_score": 0.0,
      "weight": 0.15,
      "contribution": 0.0
    },
    "ocr_injection": {
      "raw_score": 0.65,
      "weight": 0.35,
      "contribution": 0.2275
    },
    "mismatch": {
      "raw_score": 0.75,
      "weight": 0.30,
      "contribution": 0.2250
    },
    "clip": {
      "raw_score": 0.0125,
      "weight": 0.20,
      "contribution": 0.0025
    }
  },
  "hidden_text_analysis": {
    "suspicious_block_ratio": 0.0,
    "text_presence_ratio": 0.0356,
    "hidden_content_risk": 0.0,
    "has_embedded_text": false,
    "verdict": "CLEAN"
  },
  "injection_scan": {
    "combined_keywords": ["ignore"],
    "injection_risk": 0.65,
    "verdict": "INJECTION DETECTED",
    "source": "ocr"
  },
  "audio_visual_mismatch": {
    "audio_transcript": "Good morning team, let us review the quarterly metrics.",
    "image_caption": "a slide describing network architecture diagrams",
    "similarity_score": 0.1820,
    "mismatch_risk": 0.75,
    "risk_level": "HIGH",
    "note": "Significant mismatch — likely manipulation"
  },
  "clip_alignment": {
    "score": 0.9875,
    "status": "OK",
    "note": "High score = audio and image are consistent"
  },
  "processing_time_sec": 1.482
}
```

---

## 5. Dedicated FastAPI Microservice Endpoints

When running `modules/multimodal_injection/main.py` on port `8001`:

- `GET /health` -> `{"status": "running", "system": "multimodal injection detector"}`
- `POST /detect/image` -> Takes `image` form-data upload.
- `POST /detect/audio` -> Takes `audio` form-data upload.
- `POST /detect/multimodal` -> Takes `image` and/or `audio` form-data uploads.
