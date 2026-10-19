# Documentation Audit & Migration Log

> In this page: Complete audit of documentation vs codebase discrepancies, migration mapping of old files to new locations, unverifiable claims, and suggested follow-ups.  
> Status: Completed  
> Date: 2026-10-19

---

## 1. Migration Map (Old Paths -> New Paths)

Every piece of technical information from legacy documents has been preserved and migrated into the structured `/docs` hierarchy:

| Legacy Source File | Content Migrated To | Notes & Actions Taken |
| :--- | :--- | :--- |
| `/README.md` | `/README.md` (Completely rewritten) | Replaced minimal Flask stub with world-class open-source landing page. |
| `/docs/DEEPFAKE_PIPELINE_PLAN.md` | `docs/concepts/how-detection-works.md`, `docs/architecture/decisions/ADR-0003-threshold-calibration-and-decision-logic.md`, `docs/project/roadmap.md` | Absorbed serving flow, normalization mode, calibration plan, and TTA roadmap into formal docs. |
| `/docs/Deep-Fake-Detection-.../README.md` | `docs/concepts/overview.md`, `docs/models/image-detector.md`, `docs/data/datasets.md`, `docs/evaluation/results.md`, `docs/training/experiments.md` | Absorbed research overview, dataset links, benchmark table, and XAI suite into respective chapters. |
| `/docs/Deep-Fake-Detection-.../Dataset/dataset_info` | `docs/data/datasets.md` | Absorbed 140k faces dataset origin, Flickr/DFDC sources, and resolution info. |
| `/docs/Deep-Fake-Detection-.../Deepfake Voice Detection/Dataset/dataset_info` | `docs/data/datasets.md`, `docs/models/audio-detector.md` | Absorbed Deep Voice dataset description, Kaggle links, and `RandomOverSampler` usage. |
| `/docs/Deep-Fake-Detection-.../src/*.png` | `docs/assets/*.png` | Copied all 9 research visualization plots to clean assets directory and embedded into `results.md`. |
| `/modules/audio_spoof_system/README.md` | `docs/models/audio-detector.md`, `docs/guides/detecting-audio.md`, `docs/architecture/decisions/ADR-0002...md` | Absorbed Log-Mel + LFCC feature engineering, CNN/LCNN architecture, and risk scoring. |
| `/modules/multimodal_injection/README.md` | `docs/concepts/overview.md`, `docs/concepts/how-detection-works.md`, `docs/api/reference.md`, `docs/architecture/decisions/ADR-0004...md` | Absorbed multimodal injection flow, from-scratch components, and foundation model roles. |
| `/docs/demo-content.html` | Flagged in Audit (Section 2) | Identified as an unused HTMX prototype stub; preserved without deletion. |

---

## 2. Documentation Audit: Code vs Old Documentation Discrepancies

During our comprehensive audit of code, configuration files, serialized model binaries, and Jupyter notebook output streams, several discrepancies were identified between legacy documentation and the actual implementation:

### 1. DenseNet-121 Test Accuracy Discrepancy
- **Legacy Document Claim**: The table in `docs/.../README.md` reported DenseNet-121 with an accuracy of **96.8%**, precision **97.2%**, recall **96.0%**, F1 **96.6%**, and ROC AUC **0.9924**.
- **Source of Truth (Notebook Output)**: In `docs/.../densenet.ipynb` (Cell 14 output stream), the exact test evaluation on 20,000 images recorded:
  - Accuracy: **92.0%** (18,400 / 20,000)
  - Fake (0): Precision $1.00$, Recall $0.83$, F1 $0.91$
  - Real (1): Precision $0.86$, Recall $1.00$, F1 $0.92$
  - ROC AUC: **0.9924** (matches)
  - Average Precision: **0.9915**
- **Resolution**: Updated `docs/evaluation/results.md` to quote the true recorded $92.0\%$ accuracy and documented the discrepancy.

### 2. VGGFace16 Test Accuracy Discrepancy
- **Legacy Document Claim**: Reported VGGFace16 with an accuracy of **95.2%**, precision **96.0%**, recall **94.0%**, and F1 **94.8%**.
- **Source of Truth (Notebook Output)**: In `docs/.../vggface (2).ipynb` (Cell 21 output stream), evaluation on $21,591$ images recorded:
  - Accuracy: **95.0%** ($20,446 / 21,591$)
  - Fake (0): Precision $0.97$, Recall $0.92$, F1 $0.94$
  - Real (1): Precision $0.93$, Recall $0.97$, F1 $0.95$
  - ROC AUC: **0.9601**
- **Resolution**: Updated `docs/evaluation/results.md` with exact notebook values.

### 3. Production Model Architecture
- **Legacy Document Focus**: Research writeups focused almost entirely on DenseNet-121, Custom CNN, and VGGFace.
- **Source of Truth (Code & Serialized Weights)**: Direct binary inspection of the shipped production models (`models/visual_model.h5` and `models/new_model.h5`) revealed that both are 181-layer **ResNet50** models with custom dense classification heads.
- **Resolution**: Documented ResNet50 as the primary production model in `docs/models/model-cards.md` and `docs/models/image-detector.md`, while maintaining Custom CNN, DenseNet, and VGGFace in the research and experiments chapters.

### 4. Classification Threshold
- **Legacy Document Implication**: Old docs and pipeline sketches repeatedly referenced a default threshold of $0.50$.
- **Source of Truth (Code & Config)**: In `model_runtime_config.json`, the active operational threshold is calibrated to **`0.01`**.
- **Resolution**: Documented the calibrated $\tau = 0.01$ threshold in `docs/concepts/how-detection-works.md` and created [ADR-0003](architecture/decisions/ADR-0003-threshold-calibration-and-decision-logic.md).

### 5. Acoustic Pipeline Capabilities
- **Legacy Document Description**: Described the audio spoof system as a simple CNN + LCNN ensemble.
- **Source of Truth (Code)**: Inspection of `modules/audio_spoof_system/inference.py` revealed a sophisticated disagreement-aware secondary calibrator (`wav2vec2-large-xlsr`), extreme score softening logic, and risk tier assignments (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **Resolution**: Documented the full calibration fusion and risk tier architecture in `docs/models/audio-detector.md`.

---

## 3. Unverifiable Claims & TODO Items

The following claims appear in existing documents or frontend assets but cannot be fully verified from repository artifacts alone:

| Item / Claim | Location | Status | Action Taken |
| :--- | :--- | :--- | :--- |
| **"99.2% Transcription accuracy"** | `index.html` hero stat | *Reported, not reproducible from repository artifacts* | Labeled as reported marketing metric in `docs/architecture/frontend.md`. |
| **"< 3s Average processing time"** | `index.html` hero stat | *Partially Verified* | Verified for CPU visual/audio inference ($< 100\text{ms}$); multimodal inference averages $\sim 1.5–2.8\text{s}$ on CPU. |
| **"12 File formats"** | `index.html` hero stat | *Verified* | Confirmed supported extensions across image (`.jpg,.jpeg,.png,.webp,.bmp`) and audio (`.mp3,.wav,.ogg,.m4a,.flac`). |
| **Training script for `visual_model.h5`** | `models/visual_model.h5` | *TODO: verify* | The HDF5 model contains a ResNet50 topology, but the exact `.py` training script that produced `visual_model.h5` is not present in the repo (only MobileNet, Custom CNN, DenseNet, and VGGFace scripts exist). |
| **Training script for `cnn_best_clean.pth`** | `modules/audio_spoof_system/models/` | *TODO: verify* | Checkpoint weights are shipped in PyTorch, but the training script that created `cnn_best_clean.pth` and `lcnn_best_clean.pth` is missing from the repo (only an experimental Keras Colab script exists). |

---

## 4. Suggested Follow-ups for Maintainers

1. **Add an Open-Source License File**: The repository currently lacks a top-level `LICENSE` file. A standard permissive license (such as Apache 2.0 or MIT) should be committed to clarify intellectual property and distribution terms.
2. **Commit Acoustic Model Training Scripts**: The PyTorch training pipelines for `cnn_best_clean.pth` and `lcnn_best_clean.pth` should be added to `modules/audio_spoof_system/training/` to ensure end-to-end reproducibility.
3. **Move Heavy Weights to Git LFS or Hugging Face Hub**: The repository currently tracks large binary model files directly in git (`models/visual_model.h5` is $290\text{ MB}$, `new_model.h5` is $97\text{ MB}$). Moving these to Git LFS or publishing them as Hugging Face Hub release assets will keep git clone operations lightweight.
4. **Implement Continuous Integration (CI)**: Add a GitHub Actions workflow (`.github/workflows/ci.yml`) to run `scripts/import_test.py` and API smoke tests on every pull request.
5. **Upstream Face Detection**: Incorporate an automated face detector (e.g. MediaPipe Face Detector) into `preprocess_image()` to automatically crop and align faces prior to ResNet inference.
