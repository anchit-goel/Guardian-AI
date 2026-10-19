# Getting Started: Installation & Environment Setup

> In this page: Environment requirements, Python 3.10 pin, dependency installation, model weight locations, and operating system notes.  
> Who it's for: Developers setting up a local or server environment.

---

## System Requirements

- **Operating System**: macOS, Linux (Ubuntu 20.04+ recommended), or Windows 10/11 (PowerShell / WSL2).
- **Python Version**: **Python 3.10.x** (Strictly pinned to `3.10.13` in `runtime.txt` and `render.yaml`).
- **RAM**:
  - Minimum for Visual + Audio: $2\text{ GB}$
  - Recommended for Full Multimodal (Whisper + BLIP + CLIP): $4\text{ GB}+$
- **Disk Space**: $\sim 3\text{ GB}$ for Python packages; $\sim 3\text{ GB}$ for cached model weights.
- **Hardware Acceleration**: CPU inference supported out of the box; NVIDIA GPU (CUDA 12.x) supported automatically by PyTorch and TensorFlow if drivers are present.

---

## 1. Environment Creation (macOS & Linux)

### Step 1: Create a Python 3.10 Virtual Environment
```bash
# Verify Python version
python3.10 --version

# Create virtual environment
python3.10 -m venv .venv

# Activate environment
source .venv/bin/activate
```

### Step 2: Install Project Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> [!IMPORTANT]
> **NumPy Compatibility**:
> `requirements.txt` specifies `numpy<2.0`. TensorFlow 2.16 and PyTorch audio transforms are compiled against the NumPy 1.x C-API. Upgrading to NumPy 2.x will trigger ABI incompatibility warnings or segmentation faults.

---

## 2. Windows Environment Setup

On native Windows environments, use PowerShell or Command Prompt:

```powershell
# In PowerShell:
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```
Alternatively, execute the bundled batch script:
```cmd
run_flask_venv.bat
```

---

## 3. Model Weight Verification & Layout

Verify that the required model weights exist in their designated repository paths:

```text
GuardianAI/
├── models/
│   ├── visual_model.h5          # Shipped ResNet50 production model (290 MB)
│   └── new_model.h5             # Alternative visual checkpoint (97 MB)
│
└── modules/
    └── audio_spoof_system/
        └── models/
            ├── cnn_best_clean.pth   # Log-Mel SE-CNN model (402 KB)
            └── lcnn_best_clean.pth  # LFCC LCNN-MFM model (564 KB)
```

> [!NOTE]
> If `models/visual_model.h5` is not present and `allow_demo_model: true` is configured in `model_runtime_config.json`, the server will automatically generate an untrained demo network (`models/results/_demo_classifier.keras`) to allow API and frontend testing.

---

## 4. Pretrained Transformer Weight Caching

The first time you invoke `/predict-audio` (with calibrator enabled) or `/predict-multimodal`, Hugging Face Transformers will download the following open-source checkpoints:
- `openai/whisper-base` ($\sim 140\text{ MB}$)
- `Salesforce/blip-image-captioning-base` ($\sim 990\text{ MB}$)
- `openai/clip-vit-base-patch32` ($\sim 600\text{ MB}$)
- `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` ($\sim 1.2\text{ GB}$)

Weights are cached in `~/.cache/huggingface` (or the directory pointed to by `$TRANSFORMERS_CACHE` / `$HF_HOME`).
