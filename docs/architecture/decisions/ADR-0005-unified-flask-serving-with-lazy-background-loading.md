# ADR-0005: Unified Serving Gateway with Asynchronous Background Initialization

> In this page: Architecture Decision Record for asynchronous model loading and single-worker multi-threaded Gunicorn serving.  
> Status: Accepted  
> Deciders: Guardian AI Infrastructure Team  
> Date: 2026-04-10

---

## Context
Deploying deep learning microservices to PaaS providers (e.g., Render, Fly.io) involves strict startup constraints:
- Container boot scripts must bind to the assigned `$PORT` and respond to health check probes within a limited window (typically 60–120 seconds).
- Loading multiple deep learning models sequentially (ResNet50 in Keras, CNN + LCNN in PyTorch, EasyOCR, Whisper, BLIP, and CLIP) takes between 60 and 180 seconds on CPU instances, causing deploy scripts to timeout and trigger rollback.
- Multi-process WSGI managers (`gunicorn --workers 4`) replicate the Python heap across four processes, multiplying memory usage ($4 \times 1.8\text{ GB} \approx 7.2\text{ GB}$), instantly triggering Out-Of-Memory (`OOMKilled`) crashes on standard cloud tiers.

## Decision
We implemented a two-part serving strategy in `app.py` and `render.yaml`:
1. **Asynchronous Background Initialization**:
   - The Flask app binds immediately to `$PORT` and serves `GET /` and `GET /model-status` without delay.
   - Heavy multimodal models are initialized in a background daemon thread (`start_multimodal_background_load()`) protected by a thread lock (`MULTIMODAL_LOADING_LOCK`).
   - If a client requests `POST /predict-multimodal` prior to completion, the endpoint gracefully returns `HTTP 503 Service Unavailable` with a retry advice message.
2. **Single-Worker Multi-Threaded Gunicorn Architecture**:
   - Production serving is pinned to `gunicorn app:app --workers 1 --threads 4 --timeout 900`.
   - All 4 threads share a single copy of model weights in process memory, capping RAM usage under $2\text{ GB}$.

## Alternatives Considered
- **Synchronous Import-Time Loading**:
  - *Pros*: All models guaranteed ready before the first request arrives.
  - *Cons*: Guaranteed deployment failure on cloud platforms due to health check timeouts; server crashes if any single model download is interrupted.
- **Process-Based Multi-Worker (`--workers 4`)**:
  - *Pros*: True multicore parallelism bypassing the Python GIL.
  - *Cons*: Quadruples RAM requirements; requires $8\text{ GB}+$ cloud instances; prohibitive operational cost.

## Consequences & Trade-offs
### Positive Consequences:
- **Instant Health Checks**: Passes cloud platform health checks in $< 2\text{ seconds}$ from container launch.
- **Low Memory Footprint**: Entire multimodal system comfortably runs within a $2\text{ GB}$ standard cloud instance.
- **Thread Safety**: Model loading status is queryable at `GET /model-status` (`multimodal_model_loading: true`).

### Negative Consequences / Trade-offs:
- **Initial Cold-Start Window**: The first multimodal inference request within the initial 60–90 seconds may receive a 503 response until the background thread completes.
- **Python GIL Contention**: CPU inference on multiple concurrent threads is constrained by Python's Global Interpreter Lock, although underlying C++ routines in TensorFlow and PyTorch release the GIL during matrix operations.
- **Inferred Rationale**: [Inferred] This configuration was directly shaped by the Render deployment requirements evidenced in `render.yaml` (`--workers 1 --threads 4 --timeout 900`).
