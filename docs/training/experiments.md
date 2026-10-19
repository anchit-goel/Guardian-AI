# Training: Experiments & Run Log

> In this page: Documented training experiments, architecture ablations, training-efficiency mechanisms, and convergence comparisons recorded across project notebooks.  
> Who it's for: ML researchers, model developers, and experimental forensic evaluators.

---

## Experimental Runs Inventory

The repository contains multiple recorded experimental runs across diverse visual backbones and acoustic architectures:

| Run / Notebook Identifier | Architecture | Epochs / Steps | Optimizer & LR | Test Split Size | Recorded Test Accuracy | Recorded ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Model/DeepFake-CNN-Custom.ipynb` | Custom 5-stage CNN | 8 epochs (3125 steps/ep) | Adam (Warmup 0.003 -> 0.001) | 20,000 images | **95.0%** | **0.9956** |
| `Model/XAI_implementation_finalized_4epoch_3-3.ipynb` | Custom 5-stage CNN | 4 epochs (3125 steps/ep) | Adam (0.0001) | 20,000 images | 86.0% | 0.9381 |
| `custom_cnn (2).ipynb` | Custom 5-stage CNN | 10 epochs (625 steps/ep) | Adam (Default) | 20,000 images | 91.0% | 0.9858 |
| `densenet.ipynb` | DenseNet-121 + GAP + Dense(1) | 10 epochs (1562 steps/ep) | Adam (Default) | 20,000 images | 92.0% | 0.9924 |
| `vggface (2).ipynb` | VGGFace16 + GAP + Dense(1) | 5 epochs (1562 steps/ep) | Adam (Default) | 21,591 images | 95.0% | 0.9601 |
| `Deepfake_detection_hybrid_model_structure.ipynb` | Hybrid CNN Architecture | 4 epochs (3125 steps/ep) | Adam (0.0001) | 20,000 images | 86.0% | 0.9401 |
| `deep_fake_voice_recognition.py` (Colab) | Speech MLP / Transformer / LSTM | 50–100 epochs (batch 2–32) | Adam (1e-4) | 20% test split | ~92–98% train/val | N/A (Colab outputs) |

---

## 1. Custom CNN: Epoch Length & Learning Rate Ablations

Three distinct runs evaluated the 5-stage Custom CNN architecture:

### Run A: 8 Epochs with Learning Rate Warmup (`DeepFake-CNN-Custom.ipynb`)
- **Setup**: Full 100,000 training images per epoch (3125 steps $\times$ batch size 32).
- **Learning Rate**: Custom warmup scheduler:
  - Epochs 1–3: $LR = 0.003$
  - Epochs 4–8: $LR = 0.001$
- **Convergence Progression**:
  - Epoch 1: Accuracy $64.63\%$, Loss $0.6298$, Val Accuracy $74.94\%$
  - Epoch 2: Accuracy $81.69\%$, Loss $0.4040$, Val Accuracy $87.10\%$
  - Epoch 8: Evaluated on test set.
- **Test Metrics**: Accuracy $95.0\%$, ROC AUC $0.9956$, Average Precision $0.9954$.

### Run B: 4 Epochs with Low Fixed Learning Rate (`XAI_implementation_finalized_4epoch_3-3.ipynb`)
- **Setup**: 3125 steps per epoch, fixed $LR = 0.0001$.
- **Result**: Underfitted relative to Run A. Accuracy reached $86.0\%$, ROC AUC $0.9381$. Used primarily as a rapid validation run for explainability heatmap verification.

### Run C: 10 Epochs with Reduced Dataset Subset (`custom_cnn (2).ipynb`)
- **Setup**: 625 steps per epoch (40,000 samples per epoch), batch size 64.
- **Result**: Reached $91.0\%$ test accuracy, ROC AUC $0.9858$.

---

## 2. Transfer Learning: DenseNet-121 vs VGGFace16

### DenseNet-121 (`densenet.ipynb`)
- **Setup**: Pretrained DenseNet-121 base with frozen convolutional feature extractors and trainable `GlobalAveragePooling2D + Dense(1, activation='sigmoid')`.
- **Training**: 10 epochs at 1562 steps/epoch ($LR = \text{Adam default}$).
- **Evaluation**: Reached $92.0\%$ test accuracy on 20,000 test images ($1.00$ precision on fake class, $0.86$ precision on real class), ROC AUC $0.9924$.

### VGGFace16 (`vggface (2).ipynb`)
- **Setup**: Pretrained VGGFace weights loaded via `keras-vggface`.
- **Training**: 5 epochs at 1562 steps/epoch ($\sim 2600\text{ seconds}$ per epoch on GPU).
- **Evaluation**: Evaluated on $21,591$ test samples. Reached $95.0\%$ accuracy ($0.97$ precision on fake, $0.93$ precision on real), ROC AUC $0.9601$.

---

## 3. Two-Phase Fine-Tuning Setup: MobileNetV2

Preserved in lines 976–1140 of `Proposed_Custom_Hybrid-CNN(Optmized).py`, a two-phase transfer learning routine was established:

1. **Phase 1: Feature Extraction (Frozen Base)**:
   - Base model: `MobileNetV2(include_top=False, weights='imagenet')`.
   - `base_model.trainable = False`.
   - Classification head: `GAP + Dropout(0.5) + Dense(1, sigmoid)`.
   - Trained for 5 epochs with Adam ($LR = 10^{-4}$).
2. **Phase 2: Fine-Tuning (Unfreezing Deep Layers)**:
   - Unfreezes top 30 layers:
     ```python
     base_model.trainable = True
     for layer in base_model.layers[:-30]:
         layer.trainable = False
     ```
   - Recompiled with lower learning rate ($LR = 10^{-5}$) to prevent catastrophic forgetting.
   - Total 20 epochs with `ReduceLROnPlateau` ($factor=0.5, patience=2$) and `EarlyStopping` ($patience=3$).

---

## 4. Acoustic Models: Speech Transformer & Hybrid LSTM

In `docs/.../Deepfake Voice Detection/deep_fake_voice_recognition.py`:
- **Dense Baseline**: Evaluated a 3-layer MLP on 40-dimensional MFCC vectors over 100 epochs with batch size 2.
- **Transformer Sequence Model**: Applied Multi-Head Self-Attention ($num\_heads=4, ff\_dim=32$) over time-sequence MFCC frames ($max\_pad\_len=174$).
- **Ensemble Model**: Combined Transformer attention with an LSTM branch ($64$ units) concatenated prior to dense classification, incorporating L2 weight regularization ($10^{-4}$).
