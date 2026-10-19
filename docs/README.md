# Guardian AI Documentation Index

Welcome to the Guardian AI documentation. This index organizes all guides, concepts, models, datasets, and architecture references.

---

## 🚀 Getting Started
- [Quickstart](getting-started/quickstart.md): Zero to first detection in minutes across image, audio, and multimodal endpoints.
- [Installation](getting-started/installation.md): Environment setup, Python 3.10 pin, dependencies, and weight placement.
- [Configuration](getting-started/configuration.md): Complete reference for `model_runtime_config.json`, environment variables, and CLI flags.
- [Troubleshooting](getting-started/troubleshooting.md): Common errors (port conflicts, model deserialization, audio codecs, and 503 states) and resolutions.

---

## 🧠 Core Concepts
- [System Overview](concepts/overview.md): What deepfakes are, what Guardian AI detects, who it's for, and glossary.
- [How Detection Works](concepts/how-detection-works.md): End-to-end pipeline mechanics, formulas, score aggregation, and calibration fusion.
- [Responsible Use & Ethics](concepts/responsible-use.md): Operational boundaries, privacy guarantees, false positive/negative trade-offs, and forensic disclaimers.

---

## 🔬 Models & Inference
- [Model Cards](models/model-cards.md): Formal Hugging Face-style model cards for visual, audio, and sensory checkpoints.
- [Visual Deepfake Detector](models/image-detector.md): Deep-dive into ResNet50 and Custom CNN backbones, preprocessing, and the full XAI suite.
- [Acoustic Spoof Detector](models/audio-detector.md): Log-Mel CNN + LFCC LCNN architectures, Max-Feature-Map, and calibration gating.
- [Inference & Performance](models/inference.md): Runtime loading, batching, memory management, latency benchmarks, and CPU/GPU placement.

---

## 📊 Data & Datasets
- [Dataset Cards](data/datasets.md): Verified dataset cards for the 140k Faces dataset, Deep Voice dataset, and benchmark samples.
- [Preprocessing & Augmentation](data/preprocessing.md): Exact transforms, normalizations, audio filterbank parameters, and augmentations.

---

## 🏋️ Training & Evaluation
- [Training Reproduction Guide](training/training-guide.md): Kaggle/Colab training walkthrough, hyperparameters, seeds, and calibration steps.
- [Experiments & Run Log](training/experiments.md): Documented experimental runs, ablations, learning rate warmup, and transfer learning comparisons.
- [Evaluation Results & Verification](evaluation/results.md): Audited benchmark metrics, classification reports, and saved research plots.
- [Evaluation Methodology](evaluation/methodology.md): Protocol rigor, split isolation, leakage risk analysis, and scope of proof.
- [Known Limitations](evaluation/limitations.md): Candid analysis of failure modes, compression sensitivity, and adversarial evasion risks.

---

## 🏛️ Architecture & Decisions
- [System Architecture](architecture/system-overview.md): C4 context and component diagrams, sequence flows, and core design principles.
- [Backend & API Service](architecture/backend.md): Structure of Flask gateway, FastAPI microservice, and robust HDF5 deserialization.
- [Frontend Web App](architecture/frontend.md): React 18 zero-build SPA architecture, state machines, and dark-mode design system.
- [Deployment & Serving](architecture/deployment.md): Gunicorn configurations, Render Blueprint (`render.yaml`), and memory management.
- **Architecture Decision Records (ADRs)**:
  - [ADR-0001: Separate Modality Pipelines](architecture/decisions/ADR-0001-separate-image-audio-multimodal-pipelines.md)
  - [ADR-0002: Dual-Feature Ensemble for Audio](architecture/decisions/ADR-0002-dual-feature-ensemble-for-audio-spoofing.md)
  - [ADR-0003: Dynamic Threshold Calibration](architecture/decisions/ADR-0003-threshold-calibration-and-decision-logic.md)
  - [ADR-0004: Multimodal Prompt Injection Engine](architecture/decisions/ADR-0004-multimodal-injection-and-cross-modal-consistency.md)
  - [ADR-0005: Asynchronous Background Initialization](architecture/decisions/ADR-0005-unified-flask-serving-with-lazy-background-loading.md)

---

## 🔌 API Reference & Practical Guides
- [API Overview](api/overview.md): Transport protocols, status codes, base URLs, and CORS.
- [API Reference](api/reference.md): Complete endpoint specifications with curl and Python examples.
- [Detecting an Image](guides/detecting-an-image.md): Walkthrough for evaluating images and interpreting raw scores.
- [Detecting Audio](guides/detecting-audio.md): Walkthrough for evaluating speech and understanding risk tiers.
- [Fine-Tuning Guide](guides/fine-tuning-your-own-model.md): How to fine-tune visual detectors on custom dataset distributions.
- [Adding a New Modality](guides/adding-a-new-modality-or-model.md): How to add a new forensic subsystem and connect it to the gateway and UI.
- [Testing & Verification](guides/testing.md): Running unit tests, generating synthetic test samples, and verifying endpoints.

---

## 📌 Project & Meta
- [Roadmap](project/roadmap.md): Completed milestones, in-progress items, and future vision.
- [FAQ](project/faq.md): Frequently asked questions.
- [References](project/references.md): Formal academic and technical citations.
- [Documentation Audit & Changelog](CHANGELOG-DOCS.md): Comprehensive documentation audit, discrepancies, and migration log.
