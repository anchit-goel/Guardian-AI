# Security Policy

> Guardian AI is a security-focused open-source project. We take vulnerability reports, privacy compliance, and model safety seriously.

---

## Reporting a Vulnerability

If you discover a security vulnerability or exploit within Guardian AI, please do **not** open a public issue on GitHub. Instead, report it privately to the maintainers:

- **Security Contact**: Anchit Goel (via GitHub Security Advisories or maintainer profile)
- **Response Timeline**: You will receive an initial response within 48 hours acknowledging receipt of the report.
- **Coordination**: We will coordinate with you on reproduction steps, patch validation, and responsible disclosure timing.

---

## Scope & Threat Model

### In Scope:
- Remote code execution (RCE) via malicious media uploads.
- Denial of Service (DoS) resulting from crafted image, audio, or video decompression bombs.
- Arbitrary file writes or path traversal during temporary file ingestion.
- Prompt injection bypasses within the multimodal scanning engine (`modules/multimodal_injection/multimodal_detector.py`).
- Exposure of sensitive environment variables or filesystem paths via API error traces.

### Out of Scope:
- Adversarial machine learning attacks that cause false positive or false negative misclassifications without software exploitation (this is an active area of forensic research; please discuss in GitHub Discussions).
- Attacks requiring physical or local terminal access to the host machine running the server.

---

## Privacy Handling of Uploaded Media

Guardian AI is architected with strict, verifiable data minimization standards:
1. **In-Memory Processing**: Visual inference (`POST /predict`) reads raw image bytes into an in-memory `io.BytesIO` buffer. The image is never written to disk or database storage.
2. **Ephemeral Audio & Compound Files**: Audio and multimodal endpoints (`/predict-audio`, `/predict-multimodal`) write temporary `.wav` and image files to `/tmp` via `tempfile.NamedTemporaryFile` strictly to interface with C++ audio decoders (`librosa`, `soundfile`).
3. **Mandatory Unlinking**: All temporary files are wrapped in strict Python `try...finally` blocks:
   ```python
   finally:
       for path in (image_path, audio_path):
           if path and os.path.exists(path):
               os.remove(path)
   ```
   Files are unlinked from disk immediately upon completion of the inference request.
4. **Zero Telemetry**: Media tensors and user queries are never transmitted to third-party tracking or external commercial AI providers.

---

## Responsible Use & Misuse Prevention

Guardian AI models are intended for defense, verification, and forensic triage. Using this project to reverse-engineer deepfake generators to evade detection, or using automated scores to defame individuals without human verification, directly violates our project principles. Consult [Responsible Use](docs/concepts/responsible-use.md) for full operational boundaries.
