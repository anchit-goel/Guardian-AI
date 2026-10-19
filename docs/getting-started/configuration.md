# Getting Started: Configuration Reference

> In this page: Complete reference for every environment variable, configuration file key, and CLI parameter used across the Guardian AI repository.  
> Who it's for: System administrators, DevOps engineers, and deployment automators.

---

## 1. Runtime Configuration File (`model_runtime_config.json`)

The primary runtime behavior of the visual pipeline is managed by `model_runtime_config.json`, located in the workspace root or specified via the `MODEL_CONFIG_PATH` environment variable.

```json
{
  "selected_model": "models/visual_model.h5",
  "allow_demo_model": true,
  "preprocess_mode": "rescale_255",
  "threshold": 0.01,
  "positive_label": "REAL",
  "negative_label": "FAKE",
  "input_size": [
    224,
    224
  ]
}
```

### Key Reference
| Key | Type | Default | Example | Description |
| :--- | :--- | :--- | :--- | :--- |
| `selected_model` | `string \| null` | `null` | `"models/visual_model.h5"` | Relative or absolute path to the `.h5` or `.keras` model. If `null`, auto-discovery locates the newest model in the workspace. |
| `allow_demo_model`| `boolean` | `true` | `false` | When `true`, automatically creates a dummy placeholder network if no weights are found on disk, allowing frontend/API testing. |
| `preprocess_mode` | `string` | `"auto"` | `"rescale_255"` | Image scaling mode: `"rescale_255"` ($/ 255.0$), `"resnet50"`, `"mobilenet_v2"`, or `"auto"` (infers from model layer names). |
| `threshold` | `float` | `0.5` | `0.01` | Decision boundary for binary classification. Scores $> \tau$ yield `positive_label`; scores $\le \tau$ yield `negative_label`. |
| `positive_label` | `string` | `"REAL"` | `"AUTHENTIC"` | Display label assigned when `raw_score > threshold`. |
| `negative_label` | `string` | `"FAKE"` | `"SYNTHETIC"` | Display label assigned when `raw_score <= threshold`. |
| `input_size` | `[int, int]` | `[224, 224]` | `[224, 224]` | Spatial height and width target for image tensor resizing. |

---

## 2. Environment Variables

Environment variables can be provided in your shell, deployment platform settings (e.g., Render Dashboard), or `.env` files:

| Variable Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | `integer` | `8080` | Port on which the Flask server binds. Overridden automatically by cloud PaaS environments (e.g. Render `$PORT`). |
| `HOST` | `string` | `0.0.0.0` | Network interface IP address on which the HTTP server listens. |
| `MODEL_CONFIG_PATH` | `string` | `./model_runtime_config.json` | Explicit filesystem path to the runtime configuration JSON file. |
| `PYTHON_VERSION` | `string` | `3.10.13` | Declares Python interpreter version on PaaS providers (e.g., Render build environment). |
| `TRANSFORMERS_CACHE`| `string` | `~/.cache/huggingface` | Filesystem path where Hugging Face downloads and caches model weights. |
| `HF_HOME` | `string` | `~/.cache/huggingface` | Base directory for Hugging Face assets and token configurations. |

---

## 3. CLI Script Arguments

### Threshold Calibration (`scripts/calibrate_threshold.py`)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--csv` | `string` | *(Required)* | Path to calibration CSV with columns: `score,true_label`. |
| `--out` | `string` | `model_runtime_config.json` | Path to runtime configuration JSON file to update with the optimal threshold. |

### Local Score Generation (`scripts/generate_scores_local.py`)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--real-dir` | `string` | *(Required)* | Directory containing authentic ground-truth images. |
| `--fake-dir` | `string` | *(Required)* | Directory containing synthetic / deepfake images. |
| `--out` | `string` | `calibration_scores.csv` | Output destination for generated CSV scores. |
| `--config` | `string` | `model_runtime_config.json` | Configuration file containing model path and preprocessing mode. |
| `--max-per-class` | `integer` | `300` | Sample limit per class ($0$ to process all samples). |
| `--batch-size` | `integer` | `64` | Tensor inference batch size. |

### Remote API Score Generation (`scripts/generate_scores_csv.py`)
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--real-dir` | `string` | *(Required)* | Folder of real images. |
| `--fake-dir` | `string` | *(Required)* | Folder of fake images. |
| `--out` | `string` | `calibration_scores.csv` | Output CSV path. |
| `--api` | `string` | `http://127.0.0.1:8080/predict`| Live HTTP endpoint to query via multipart form-data. |
| `--max-per-class` | `integer` | `0` | Sample limit per class ($0$ for all). |
