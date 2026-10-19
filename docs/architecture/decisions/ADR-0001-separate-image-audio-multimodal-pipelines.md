# ADR-0001: Separate Image, Audio, and Multimodal Forensic Pipelines

> In this page: Architecture Decision Record for decoupling visual detection, acoustic anti-spoofing, and multimodal consistency into dedicated pipelines.  
> Status: Accepted  
> Deciders: Guardian AI Engineering Team  
> Date: 2026-04-10

---

## Context
Deepfake and manipulated media manifests across different sensory modalities: facial image manipulation (e.g. StyleGAN artifacts, face swaps), synthetic speech generation (e.g. Tacotron TTS, voice conversion), and cross-modal deceptive packaging (e.g. authentic audio over an unrelated scene or embedded prompt injection).
We needed to decide whether to train and serve a single end-to-end multimodal foundation model (such as a unified vision-audio transformer) or decouple detection into distinct, specialized pipelines.

## Decision
We decided to architect Guardian AI with three independent, modular forensic subsystems:
1. An isolated visual pipeline (`POST /predict`) powered by CNN classifiers (ResNet50 / Custom CNN).
2. An isolated acoustic pipeline (`POST /predict-audio`) powered by Log-Mel CNN + LFCC LCNN models.
3. A compound multimodal pipeline (`POST /predict-multimodal`) that invokes vision, speech, OCR, and alignment extractors dynamically.

## Alternatives Considered
- **Single Unified Vision-Audio Foundation Model (e.g., Audio-Visual Transformer)**:
  - *Pros*: Potential joint representation learning; single forward pass for video files.
  - *Cons*: Immense training data requirements; cannot evaluate single image or audio-only files efficiently; massive latency and memory footprint during deployment; catastrophic failure if one modality is absent.
- **Microservices across Separate Repositories**:
  - *Pros*: Complete dependency isolation.
  - *Cons*: Operational overhead, complex orchestration for simple deployments, fragmented developer experience.

## Consequences & Trade-offs
### Positive Consequences:
- **Modality Agnostic**: Clients can send an image alone, an audio clip alone, or both without dummy padding or simulated inputs.
- **Independent Evolution**: The audio team can optimize LFCC features and retrain PyTorch models without affecting the Keras visual model.
- **Resource Efficiency**: Image-only requests consume only Keras/TensorFlow resources without allocating memory for speech or OCR pipelines.

### Negative Consequences / Trade-offs:
- **Cross-Framework Overhead**: The project depends on both TensorFlow/Keras and PyTorch/Transformers simultaneously, increasing virtual environment size (~3GB+ total dependencies).
- **Inferred Rationale**: [Inferred] Maintaining separate frameworks was chosen to leverage existing specialized research codebases (Keras for facial CNNs, PyTorch for audio anti-spoofing) rather than undertaking a full rewrite into a single framework.
