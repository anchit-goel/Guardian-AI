# Evaluation: Methodology & Rigor Analysis

> In this page: Evaluation protocol details, train/val/test split methodology, data leakage risks, and rigorous assessment of what the evaluation numbers do and do not prove.  
> Who it's for: ML researchers, benchmark reviewers, and academic peer-reviewers.

---

## Evaluation Protocol

Across the visual detection notebooks in the repository, models were evaluated using standard classification metrics:
1. **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**: Measures ranking quality across all classification thresholds, independent of class distribution.
2. **Average Precision (AP)**: Summarizes the Precision-Recall curve, particularly informative when evaluating false alarm rates.
3. **Macro and Per-Class F1-Scores**: Geometric mean of precision and recall for both Authentic (1) and Synthetic (0) classes.

---

## 1. Split Separation & Independence

In the primary Kaggle visual training workflows:
- **Partitioning**: The combined dataset (`140k Real & Fake Faces` + `Real & Fake Face Detection`) was split into dedicated directories:
  - `real_vs_fake/real-vs-fake/train/` ($\sim 100,000$ samples)
  - `real_vs_fake/real-vs-fake/valid/` ($\sim 20,000$ samples)
  - `real_vs_fake/real-vs-fake/test/` ($20,000$ samples)
- **Generator Isolation**: `test_flow` was initialized with `shuffle=False` and `batch_size=1` to guarantee deterministic evaluation order and strict exclusion from gradient backpropagation:
  ```python
  test_flow = image_gen1.flow_from_directory(
      base_path + 'test/',
      target_size=(224, 224),
      batch_size=1,
      shuffle=False,
      class_mode='binary'
  )
  ```

---

## 2. Leakage Risk Analysis

Forensic evaluation on synthetic facial datasets is prone to subtle data leakage vectors:

### A. Generator Fingerprint Overfitting
- **The Risk**: If synthetic faces in both the train and test sets are generated using the same StyleGAN checkpoint, the neural network may learn the specific checkerboard upsampling artifacts or convolutional kernel fingerprints of that generator rather than generalized deepfake manipulation cues.
- **Observed Behavior**: The near-perfect ROC AUC of $0.9956$ on the test set strongly indicates that the model has fully captured the StyleGAN generator manifold. However, performance will degrade when tested against unseen generative architectures (such as Latent Diffusion Models).

### B. Identity Leakage in Video-Extracted Frames
- **The Risk**: When datasets derive face crops from video sequences (such as the Facebook DFDC challenge), multiple frames from the same subject video clip may inadvertently leak across train and test splits.
- **Mitigation in Repository**: The primary visual dataset is constructed from FFHQ and 1 Million Fake Faces, which are static single-image portrait generations rather than continuous video streams, eliminating adjacent video frame correlation.

---

## 3. What the Numbers Prove & What They Do Not

### What the Numbers Prove:
- **High In-Domain Discriminability**: On clean $256 \times 256$ portrait crops, the Custom CNN and ResNet models achieve $\ge 95\%$ accuracy and $\ge 0.99$ ROC AUC in distinguishing authentic human portraits from StyleGAN synthetic faces.
- **Low False Alarm Rate on Clean Imagery**: Class 1 (Real) precision reached $0.99$ in the Custom CNN, demonstrating that clean, unmanipulated photos are rarely misclassified as fakes within this domain.

### What the Numbers Do NOT Prove:
- **Zero-Shot Diffusion Transfer**: These numbers do not prove high accuracy against modern text-to-image diffusion models (Midjourney, DALL-E 3, FLUX), which lack the specific frequency artifacts of early GANs.
- **Social Media Invariance**: These numbers do not reflect performance on images re-compressed by social media messaging apps (JPEG compression factor $< 70$, downsampling to $< 100\text{px}$), which attenuates the high-frequency pixel cues the classifier relies on.
- **Forensic Legal Infallibility**: A high test score on a benchmark does not constitute judicial or biometric certainty in unconstrained real-world settings.
