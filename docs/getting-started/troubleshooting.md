# Getting Started: Troubleshooting

> In this page: Common errors, log messages, hardware conflicts, and exact resolutions derived directly from the codebase.  
> Who it's for: Operators, developers, and support engineers encountering runtime errors.

---

## 1. Network & Port Conflicts

### Error: `Address already in use: Port 8080`
**Symptom**: When running `python app.py` or Gunicorn, the process exits with:
`OSError: [Errno 48] Address already in use`

**Cause**: A previous instance of `app.py` or another local daemon is already listening on port 8080.

**Resolution**:
Find and terminate the lingering process:
```bash
# On macOS / Linux:
lsof -i :8080
kill -9 <PID>

# Alternatively, run on a different port:
PORT=8081 python app.py
```

---

## 2. Model Loading & Weights Issues

### Error: `No model weights found`
**Symptom**:
`GET /model-status` returns `model_loaded: false` and `error: No model weights found.`

**Cause**:
No `.h5` or `.keras` file was found under `models/`, and `allow_demo_model` is set to `false` in `model_runtime_config.json`.

**Resolution**:
1. Ensure your model weight file (e.g. `visual_model.h5`) is located inside the `models/` directory.
2. In `model_runtime_config.json`, verify that `selected_model` points to the exact path (`"selected_model": "models/visual_model.h5"`).
3. If you only want to test the API and frontend without real weights, set:
   ```json
   "allow_demo_model": true
   ```
   The backend will generate a tiny randomly-initialized placeholder model (`_demo_classifier.keras`).

---

### Error: `DepthwiseConv2D got unexpected keyword argument 'groups'`
**Symptom**: Keras crashes when attempting to deserialize legacy `.h5` files:
`TypeError: DepthwiseConv2D.__init__() got an unexpected keyword argument 'groups'`

**Cause**: An architectural schema incompatibility between older Keras 2 models and Keras 3.

**Resolution in Code**:
Guardian AI includes an automatic compatibility layer (`CompatibleDepthwiseConv2D` and `load_model_robust()` in `app.py`) that strips the invalid `groups` attribute directly from the serialized JSON attributes before instantiating the layer. If you are writing a custom loader script, ensure you import the compatibility wrapper:
```python
from app import CompatibleDepthwiseConv2D
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D}
)
```

---

## 3. Multimodal & Audio Processing Issues

### Error: `HTTP 503: Multimodal model is still loading`
**Symptom**: `POST /predict-multimodal` returns `HTTP 503 Service Unavailable`:
`{"error": "Multimodal model is still loading. Please try again in a minute."}`

**Cause**: Heavy multimodal transformer checkpoints (Whisper, BLIP, CLIP) are being downloaded or initialized on an asynchronous background thread.

**Resolution**:
Check initialization status at `GET /model-status`:
- If `multimodal_model_loading: true`, wait 30–60 seconds.
- Once `multimodal_model_loaded: true`, re-send the inference request.

---

### Error: Audio Format or Codec Failure (`soundfile` / `librosa`)
**Symptom**: `POST /predict-audio` fails with:
`Error opening file ...: Format not recognised.` or `No backend available.`

**Cause**: The uploaded audio file uses a proprietary or compressed container (such as AAC or `.m4a`) and the host system lacks `ffmpeg` to decode non-PCM audio streams.

**Resolution**:
Install `ffmpeg` on your host operating system:
```bash
# On macOS:
brew install ffmpeg

# On Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## 4. Cloud Deployment (Render) Failures

### Issue: TensorFlow Build Fails on Render
**Symptom**: Render build logs fail during wheel installation with:
`ERROR: Could not find a version that satisfies the requirement tensorflow==2.16.2`

**Cause**: Render defaulted to Python 3.11 or 3.12, where TensorFlow 2.16 wheels were not yet published for that platform.

**Resolution**:
Ensure Render is using Python 3.10.x. In the Render Dashboard under **Environment**:
1. Add environment variable: `PYTHON_VERSION = 3.10.13`
2. Ensure `runtime.txt` contains `python-3.10.13` (shipped in the repository).
3. Trigger a manual deploy with **Clear build cache**.
