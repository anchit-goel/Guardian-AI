# Models: Model Cards

> In this page: Formal Hugging Face-style model cards for all primary, secondary, and sensory models used in Guardian AI.  
> Who it's for: ML engineers, model auditors, compliance reviewers, and AI researchers.

---

## Model Card Index

1. [Guardian-Visual-ResNet50 (Production Visual Classifier)](#1-guardian-visual-resnet50)
2. [Guardian-Visual-CustomCNN (Research Baseline)](#2-guardian-visual-customcnn)
3. [Guardian-Audio-CNN-SE (Primary Acoustic Mel Detector)](#3-guardian-audio-cnn-se)
4. [Guardian-Audio-LCNN-MFM (Primary Acoustic LFCC Detector)](#4-guardian-audio-lcnn-mfm)
5. [Sensory Foundation Checkpoints](#5-sensory-foundation-checkpoints)

---

## 1. Guardian-Visual-ResNet50

### Model Details
- **Architecture**: 181-layer Deep Residual Network (ResNet50) with custom multi-layer classification head.
- **Model Checkpoints**: `models/visual_model.h5` (290 MB) and `models/new_model.h5` (97 MB).
- **Task**: Binary image classification (Synthetic/Manipulated vs Real).
- **Framework**: TensorFlow 2.16.2 / Keras.
- **Base Checkpoint License**: Apache 2.0 (Keras Applications ImageNet pre-training).

### Model Architecture Topology
```text
Input (224, 224, 3)
  │
ResNet50 Backbone (Layers: conv1_pad through conv5_block3_out)
  │
GlobalAveragePooling2D
  │
Dense (256 units, ReLU activation)
  │
Dropout (rate = 0.5)
  │
Dense (128 units, ReLU activation)
  │
Dropout (rate = 0.3)
  │
Dense (1 unit, Sigmoid activation) -> Output ∈ [0, 1]
```

### Inputs & Preprocessing
- **Modality**: RGB Image (`.jpg`, `.png`, `.webp`, `.bmp`).
- **Resolution**: $224 \times 224$ pixels (bilinear resize).
- **Channels**: 3 (RGB order).
- **Scaling**: Pixel intensity divided by $255.0$ (`rescale_255`) yielding floats in $[0.0, 1.0]$.

### Training Data & Procedure
- **Dataset**: 140K Real and Fake Faces combined with Real and Fake Face Detection (142,356 balanced portrait images: 71,178 real from Flickr/DFDC, 71,178 synthetic from StyleGAN / 1M Fake Faces).
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$).
- **Loss Function**: Binary Cross-Entropy.

### Limitations & Biases
- **Domain Focus**: Trained strictly on aligned human portrait faces; unreliable on full-body photos, architectural scenes, or animal imagery.
- **Generation Artifacts**: Highly tuned to GAN checkerboard and boundary artifacts; lower recall on ultra-recent photorealistic diffusion architectures (e.g. Midjourney v6, FLUX).

---

## 2. Guardian-Visual-CustomCNN

### Model Details
- **Architecture**: 5-stage Custom Convolutional Neural Network with Batch Normalization and Dropout.
- **Reference Script / Notebook**: `Proposed_Custom_Hybrid-CNN(Optmized).py`, `Model/DeepFake-CNN-Custom.ipynb`.
- **Task**: Binary facial deepfake classification.
- **Framework**: Keras / TensorFlow.

### Architecture Topology
- **Stage 1**: `Conv2D(16, 3x3, same) + MaxPool(2x2) + BatchNorm`
- **Stage 2**: `Conv2D(32, 3x3, same) + MaxPool(2x2) + BatchNorm + Dropout(0.1)`
- **Stage 3**: `Conv2D(64, 3x3, same) + MaxPool(2x2) + BatchNorm + Dropout(0.1)`
- **Stage 4**: `Conv2D(128, 3x3, same) + MaxPool(2x2) + BatchNorm + Dropout(0.1)`
- **Stage 5**: `Conv2D(256, 3x3, same) + MaxPool(2x2) + BatchNorm + Dropout(0.1)`
- **Head**: `GlobalAveragePooling2D + Dense(1, sigmoid)`

### Recorded Benchmark Performance
*(Recorded directly in `Model/DeepFake-CNN-Custom.ipynb` outputs on 20,000-image test set)*
- **ROC AUC**: $0.9956$
- **Average Precision (AP)**: $0.9954$
- **Accuracy**: $95.0\%$
- **Fake Class (0)**: Precision $0.92$, Recall $0.99$, F1 $0.95$
- **Real Class (1)**: Precision $0.99$, Recall $0.92$, F1 $0.95$

---

## 3. Guardian-Audio-CNN-SE

### Model Details
- **Architecture**: 3-stage 2D Convolutional Network with Squeeze-and-Excitation (SE) channel attention.
- **Checkpoint**: `modules/audio_spoof_system/models/cnn_best_clean.pth` ($402\text{ KB}$).
- **Task**: Voice anti-spoofing (Genuine speech vs TTS / Voice Conversion).
- **Framework**: PyTorch.

### Architecture Specifications
- **Input**: Log-Mel spectrogram tensor $(1, 1, 80, T)$.
- **Feature Extraction**: $80$ Mel bands, $N_{fft}=400$, $\text{hop}=160$, sample rate $16\text{ kHz}$.
- **Convolutions**:
  - Layer 1: $\text{Conv2D}(1 \to 32, 3\times 3) + \text{BatchNorm} + \text{ReLU} + \text{MaxPool}(2\times 2)$
  - Layer 2: $\text{Conv2D}(32 \to 64, 3\times 3) + \text{BatchNorm} + \text{ReLU} + \text{MaxPool}(2\times 2)$
  - Layer 3: $\text{Conv2D}(64 \to 128, 3\times 3) + \text{BatchNorm} + \text{ReLU} + \text{MaxPool}(2\times 2)$
- **Attention**: Squeeze-and-Excitation block on 128 channels with reduction ratio $r=8$.
- **Head**: `AdaptiveAvgPool2D(1) + Dropout(0.3) + Linear(128 -> 2)` (Raw un-normalized logits).

---

## 4. Guardian-Audio-LCNN-MFM

### Model Details
- **Architecture**: Light Convolutional Neural Network (LCNN) with Max-Feature-Map (MFM) competitive activations.
- **Checkpoint**: `modules/audio_spoof_system/models/lcnn_best_clean.pth` ($564\text{ KB}$).
- **Task**: Voice anti-spoofing on Linear Frequency Cepstral Coefficients.
- **Framework**: PyTorch.

### Architecture Specifications
- **Input**: LFCC tensor $(1, 1, 60, T)$ extracted via `torchaudio.transforms.LFCC`.
- **Convolutions with MFM**:
  - `Conv2D(1 -> 64, 5x5) + MFM() -> 32 channels + MaxPool(2x2)`
  - `Conv2D(32 -> 64, 1x1) + MFM() -> 32 channels`
  - `Conv2D(32 -> 96, 3x3) + MFM() -> 48 channels + MaxPool(2x2)`
  - `Conv2D(48 -> 96, 1x1) + MFM() -> 48 channels`
  - `Conv2D(48 -> 128, 3x3) + MFM() -> 64 channels + MaxPool(2x2)`
  - `Conv2D(64 -> 128, 1x1) + MFM() -> 64 channels`
  - `Conv2D(64 -> 64, 3x3) + MFM() -> 32 channels`
  - `Conv2D(32 -> 64, 1x1) + MFM() -> 32 channels`
- **Head**: `AdaptiveAvgPool2D(1) + Dropout(0.3) + Linear(32 -> 2)` (Raw un-normalized logits).

---

## 5. Sensory Foundation Checkpoints

Pretrained open-source checkpoints utilized strictly as non-trainable sensory extractors in `modules/multimodal_injection/multimodal_detector.py` and `modules/audio_spoof_system/pretrained_detector.py`:

| Model Identifier | Provider | Primary Role | Parameters / Size | Base License |
| :--- | :--- | :--- | :--- | :--- |
| `openai/whisper-base` | OpenAI | Audio-to-text speech transcription | $\sim 74\text{M}$ params | MIT |
| `Salesforce/blip-image-captioning-base` | Salesforce | Zero-shot visual scene captioning | $\sim 224\text{M}$ params | BSD-3-Clause |
| `openai/clip-vit-base-patch32` | OpenAI | Cross-modal text-image embedding alignment | $\sim 151\text{M}$ params | MIT |
| `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` | Hugging Face Hub | Acoustic anti-spoofing secondary calibrator | $\sim 317\text{M}$ params | Apache 2.0 |
| `EasyOCR (en)` | JaidedAI | Optical character recognition in visual scenes | CRAFT + ResNet | Apache 2.0 |
