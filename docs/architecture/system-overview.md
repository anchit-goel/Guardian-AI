# Architecture: System Overview

> In this page: System context, component architecture, data flow paths, sequence flows, and cross-modal decision boundaries.  
> Who it's for: Systems architects, backend developers, and DevOps engineers.

---

## System Context Diagram

Guardian AI acts as a unified media forensics gateway. Clients interact via an HTTP REST API or a browser-based Single Page Application (SPA).

```mermaid
C4Context
    title System Context for Guardian AI

    Person(user, "User / Security Analyst", "Submits images, audio clips, or compound media for synthetic manipulation analysis.")
    System(guardian, "Guardian AI Platform", "Flask API, React SPA, and ML Inference Engines across Visual, Acoustic, and Multimodal modalities.")
    System_Ext(hf, "Hugging Face Hub", "Provides weights for secondary sensors (Whisper, BLIP, CLIP, wav2vec2).")

    Rel(user, guardian, "Uploads media via HTTP POST / Web UI", "HTTPS / JSON")
    Rel(guardian, hf, "Downloads & caches pretrained model weights (first boot)", "HTTPS")
```

---

## Container & Component Diagram

The repository is organized into three primary operational tiers: the Web Serving Layer, the Specialized Model Execution Modules, and the Local Model Cache.

```mermaid
graph TB
    subgraph Client ["Client Layer"]
        Browser["Web Browser (index.html React SPA)"]
        CurlClient["CLI / Automation Client (curl / python requests)"]
    end

    subgraph Gateway ["Serving & Routing Layer (app.py)"]
        FlaskServer["Flask Application (Port 8080 / Gunicorn)"]
        Cors["Flask-CORS Middleware"]
        ConfigMgr["Runtime Config Manager (model_runtime_config.json)"]
        DiscEngine["Model Discovery Engine (discover_models)"]
    end

    subgraph Pipelines ["Forensic Subsystems"]
        VisualSub["Visual Detection Engine\n• ResNet50 (models/visual_model.h5)\n• Fallback Demo Classifier"]
        AudioSub["Audio Spoof System (modules/audio_spoof_system/)\n• Log-Mel CNN + SE\n• LFCC LCNN + MFM\n• wav2vec2 Secondary Calibrator"]
        MultiSub["Multimodal Engine (modules/multimodal_injection/)\n• EasyOCR + Ghost Text Scanner\n• Whisper-Base ASR\n• BLIP Captioner\n• CLIP-ViT-B/32 Alignment"]
    end

    subgraph Storage ["Filesystem & Cache"]
        ModelDir["Local Weights (/models/*.h5, *.pth)"]
        HFCache["Transformers Local Cache ($TRANSFORMERS_CACHE)"]
        TempDir["Ephemeral System Temp (/tmp)"]
    end

    Browser -->|HTTP| FlaskServer
    CurlClient -->|HTTP| FlaskServer
    FlaskServer --> Cors
    FlaskServer --> ConfigMgr
    FlaskServer --> DiscEngine

    FlaskServer -->|In-Memory Bytes| VisualSub
    FlaskServer -->|Temp Audio Path| AudioSub
    FlaskServer -->|Temp File Paths| MultiSub

    VisualSub --> ModelDir
    AudioSub --> ModelDir
    AudioSub -.-> HFCache
    MultiSub -.-> HFCache
    MultiSub --> TempDir
    AudioSub --> TempDir
```

---

## Sequence Diagram: Detection Request Lifecycle

The diagram below details the execution lifecycle when handling an audio-visual compound request to `POST /predict-multimodal`:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant App as Flask Gateway (app.py)
    participant Temp as Temp Storage (/tmp)
    participant Multi as Multimodal Detector
    participant Sensory as Sensory Extractors (Whisper/BLIP/EasyOCR)
    participant Fusion as From-Scratch Risk Scorer

    Client->>App: POST /predict-multimodal (multipart: image, audio)
    App->>Temp: Write ephemeral image & audio files
    App->>Multi: analyze_multimodal(image_path, audio_path)
    
    par Visual Branch
        Multi->>Sensory: EasyOCR(image) -> tokens
        Multi->>Sensory: BLIP(image) -> caption
        Multi->>Multi: detect_low_contrast_text(image) -> ghost text risk
    and Acoustic Branch
        Multi->>Sensory: Whisper(audio) -> transcript
    end

    Multi->>Fusion: scan_for_injection(caption, ocr_tokens, transcript)
    Multi->>Fusion: from_scratch_similarity(transcript, caption)
    Multi->>Sensory: CLIP(image, transcript) -> alignment score
    Multi->>Fusion: compute_overall_risk(hidden, ocr_inj, mismatch, clip)
    Fusion-->>Multi: Overall Risk Score, Tier, & Signal Breakdown
    Multi-->>App: Result Dictionary
    App->>Temp: os.remove(image_path), os.remove(audio_path)
    App-->>Client: 200 OK (JSON Report)
```

---

## Key Architectural Principles

1. **Decoupled Forensic Modules**: The visual, audio, and multimodal subsystems reside in isolated packages. The visual pipeline runs entirely on TensorFlow/Keras, while the audio and multimodal pipelines run on PyTorch and Hugging Face Transformers.
2. **Graceful Degradation**:
   - If heavyweight visual weights are missing, the server can generate an untrained demo classifier (`_demo_classifier.keras`) to maintain API responsiveness for local UI development if `allow_demo_model` is enabled.
   - If the audio transformer calibrator (`wav2vec2`) fails to download or runs out of memory, the audio engine automatically falls back to the primary scratch-built CNN + LCNN ensemble (`mode: primary_only`).
   - The multimodal engine loads asynchronously on a background thread during application startup (`start_multimodal_background_load()`) to prevent blocking port binding and avoid deploy timeouts on platforms like Render.
3. **Stateless Media Lifecycle**: Media is processed in-memory or written to short-lived temporary files that are guaranteed to be unlinked in `finally` clauses. No media files or biometric embeddings are persisted to disk.
