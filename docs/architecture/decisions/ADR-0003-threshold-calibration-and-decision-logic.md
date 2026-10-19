# ADR-0003: Dynamic Threshold Calibration & Decoupled Runtime Configuration

> In this page: Architecture Decision Record for externalizing classification thresholds to runtime configuration and providing offline calibration scripts.  
> Status: Accepted  
> Deciders: Guardian AI Engineering Team  
> Date: 2026-04-10

---

## Context
Standard binary classification networks output a sigmoid activation $\hat{y} \in [0, 1]$. In naive deployment architectures, a static decision threshold of $\tau = 0.50$ is hardcoded. However, in facial deepfake detection:
- Test-time data often exhibits severe domain shift compared to clean training datasets (e.g. compression artifacts, lighting, motion blur).
- Output probability distributions may shift or cluster near extreme values.
- Operational requirements differ: moderation triage requires high sensitivity (low false negative rate for fake detection), whereas automated fraud detection may require high specificity (low false positive rate).
- In the repository's `model_runtime_config.json`, the active threshold was calibrated to `0.01` rather than `0.50`.

## Decision
We decided to:
1. Externalize the decision boundary to `model_runtime_config.json` (`threshold: 0.01`), allowing operators to re-calibrate without modifying code or restarting with code deployments.
2. Provide automated calibration tooling (`scripts/calibrate_threshold.py`) that evaluates score distributions across labeled validation sets (e.g. 50 real, 50 fake) and optimizes the threshold for maximum F1-score.
3. Decouple label definitions (`positive_label: "REAL"`, `negative_label: "FAKE"`) into configuration so that downstream clients can customize display terms without altering backend logic.

## Alternatives Considered
- **Hardcoding Threshold $0.50$ in `app.py`**:
  - *Pros*: Eliminates configuration file dependencies.
  - *Cons*: Highly brittle under real-world data shift; causes catastrophic false negative rates if model output scores cluster below 0.5.
- **Platt Scaling / Temperature Scaling inside Keras Model**:
  - *Pros*: Mathematical calibration inside the computation graph.
  - *Cons*: Requires modifying and re-saving the serialized `.h5` model artifact every time calibration data shifts; cannot be changed dynamically per deployment environment.

## Consequences & Trade-offs
### Positive Consequences:
- **Zero-Downtime Tuning**: Operational teams can update `model_runtime_config.json` and restart the process to immediately adjust sensitivity.
- **Reproducible Calibration**: The exact grid-search sweep across $\tau \in [0.01, 0.99]$ is versioned in `scripts/calibrate_threshold.py`.
- **Explainability**: Both the `raw_score` and `threshold` are returned in every JSON prediction payload for auditable decision logging.

### Negative Consequences / Trade-offs:
- **Configuration Desynchronization**: If the config file is corrupted or points to a non-existent model path, the backend must maintain safe fallbacks (`load_runtime_config()` defaults to $\tau = 0.5$).
- **Inferred Rationale**: [Inferred] Setting $\tau = 0.01$ in the shipped config reflects an empirical optimization on a validation benchmark where real images received low raw scores, necessitating a low threshold to prevent high false negative rates on fake imagery.
