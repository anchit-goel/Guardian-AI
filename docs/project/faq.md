# Project: Frequently Asked Questions (FAQ)

> In this page: Answers to common questions regarding architecture, model capabilities, performance, and operational deployment.  
> Who it's for: All users, researchers, and systems integrators.

---

## 1. General & Architecture

### What is the difference between Guardian AI and other deepfake detectors?
Most open-source detectors focus exclusively on one modality (either static facial images or isolated audio clips). Guardian AI integrates three decoupled forensic pipelines under a single gateway:
1. Visual face deepfake classification (ResNet50 / Custom CNN).
2. Acoustic speech anti-spoofing (Log-Mel CNN + LFCC LCNN).
3. Cross-modal prompt injection and audio-visual consistency checking (EasyOCR + Whisper + BLIP + CLIP).

### Who built Guardian AI and what is its origin?
Guardian AI was prepared and developed as part of the **HACK IITK 2026 Grand Finale**.

**Project Maintainers**:
- **Anchit Goel**
- **Malhar Srivatsav**
- **Ryan Gupta**

### Why are both TensorFlow and PyTorch in `requirements.txt`?
The visual models were trained and serialized in Keras/TensorFlow (`.h5`), while the audio spoof models, EasyOCR, and Hugging Face Transformers (Whisper, BLIP, CLIP) are built in PyTorch. Both frameworks run side-by-side in process memory.

### Does Guardian AI transmit media to third-party cloud APIs?
**No.** All inference is performed locally or within your dedicated hosting environment. Media bytes are never sent to external commercial APIs (such as OpenAI or Google). Pretrained foundation checkpoints (Whisper, BLIP, CLIP) run entirely on local compute after being downloaded.

---

## 2. Forensic Capabilities & Limitations

### Can Guardian AI detect full video files (`.mp4`)?
Currently, the REST API accepts static images (`.jpg`, `.png`, `.webp`, `.bmp`) and audio files (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`). To analyze a video, extract keyframes as images and the audio track as a `.wav` file, then submit them to `/predict-multimodal`. Native temporal video processing is listed in our [Roadmap](roadmap.md).

### Why is the visual threshold set to `0.01` instead of `0.50`?
The threshold in `model_runtime_config.json` was empirically optimized via `scripts/calibrate_threshold.py` on a held-out validation set. On that benchmark, setting $\tau = 0.01$ maximized the F1-score and minimized false negatives (ensuring that subtle deepfakes are caught rather than missed). You can tune this threshold on your own dataset using `scripts/calibrate_threshold.py`.

### How well does it detect modern diffusion models (e.g. Midjourney, FLUX)?
The visual models were trained primarily on GAN-generated faces (StyleGAN and DFDC). While they detect facial anomalies and boundary blending well, raw confidence may be lower on ultra-photorealistic diffusion outputs. Combining the visual pipeline with the multimodal engine provides stronger cross-modal guarantees.

---

## 3. Operational & Serving

### Why does `/predict-multimodal` return `HTTP 503` right after starting the server?
Heavy transformer models (Whisper, BLIP, CLIP) are initialized asynchronously in a background thread to prevent blocking cloud health checks. If you send a request within the first 30–60 seconds of server boot, the endpoint returns a 503 retry prompt. Once initialization completes, subsequent requests execute immediately.

### Can I run Guardian AI on a GPU?
**Yes.** PyTorch and TensorFlow will automatically detect and utilize CUDA/MPS accelerators if available. However, the system is explicitly optimized for fast CPU inference ($\sim 30\text{ ms}$ for audio, $\sim 80\text{ ms}$ for image on standard modern CPUs).

### How do I customize the labels returned by the API?
Edit `model_runtime_config.json`:
```json
{
  "positive_label": "AUTHENTIC",
  "negative_label": "MANIPULATED"
}
```
Restart the server, and the API will return your preferred nomenclature.
