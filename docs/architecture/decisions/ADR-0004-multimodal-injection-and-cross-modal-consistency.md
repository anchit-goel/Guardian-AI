# ADR-0004: Multimodal Prompt Injection & Cross-Modal Consistency Engine

> In this page: Architecture Decision Record for combining handcrafted statistical detectors with pretrained foundation sensors for multimodal threat detection.  
> Status: Accepted  
> Deciders: Guardian AI Security & AI Safety Team  
> Date: 2026-04-10

---

## Context
Multimodal Artificial Intelligence systems (such as Vision-Language Models and voice-enabled conversational agents) are exposed to novel threat vectors:
1. **Visual Prompt Injection**: Malicious instructions embedded in images via low-contrast "ghost text", background micro-typography, or visual steganography to hijack downstream LLM ingest systems.
2. **Acoustic Prompt Injection**: Spoken commands embedded into background noise or music.
3. **Audio-Visual Disinformation**: Authentic audio tracks paired with completely unrelated or contradictory visual scenes to mislead fact-checkers.

We needed a system capable of detecting both explicit text injections and subtle cross-modal narrative mismatches without depending on external cloud LLM APIs.

## Decision
We engineered a hybrid architecture that pairs from-scratch statistical detection algorithms with local open-source neural sensory extractors:
- **From-Scratch Forensic Logic**:
  - Pixel-level variance analysis ($32 \times 32$ tiles) to detect low-contrast stenographic text.
  - Fuzzy sliding-window keyword scanner using `rapidfuzz` against an 18-keyword adversarial lexicon.
  - Semantic similarity scorer using a combination of Term-Frequency (TF) cosine similarity ($0.60$) and Jaccard token overlap ($0.40$).
  - Transparent linear risk aggregation with explicit signal breakdown weights ($0.15$ hidden text, $0.35$ OCR injection, $0.30$ mismatch, $0.20$ CLIP alignment).
- **Sensory Extractors (Feature Extractors Only)**:
  - `openai/whisper-base`: Converts speech into transcribed text.
  - `Salesforce/blip-image-captioning-base`: Generates factual scene descriptions from images.
  - `EasyOCR`: Extracts optical character strings from visual rasters.
  - `openai/clip-vit-base-patch32`: Measures direct image-text semantic embedding alignment.

## Alternatives Considered
- **Cloud LLM-as-a-Judge (e.g. OpenAI GPT-4V API)**:
  - *Pros*: Natural language reasoning out of the box.
  - *Cons*: Expensive per-request cost; high latency (>2–4s); violates privacy by transmitting user media to third-party providers; susceptible to the very prompt injection attacks it is supposed to detect.
- **End-to-End Deep Neural Network Classifier**:
  - *Pros*: Single forward pass.
  - *Cons*: Total black box; cannot provide explainable signal breakdowns (cannot tell the user *why* an image is risky); prone to shortcut learning and poor out-of-domain generalization.

## Consequences & Trade-offs
### Positive Consequences:
- **Full Explainability**: The output JSON provides an exact breakdown of each contributing signal (`raw_score`, `weight`, `contribution`), identified keywords, and individual token confidence values.
- **Privacy & Air-Gapped Operation**: Runs entirely on local compute without outbound internet access once model weights are cached.
- **Zero Hallucination**: The core detection logic is deterministic Python code; it cannot hallucinate nonexistent threats.

### Negative Consequences / Trade-offs:
- **Model Download Footprint**: Caching Whisper, BLIP, EasyOCR, and CLIP requires approximately $2.5\text{ GB}$ of disk space.
- **Heuristic Sensitivity**: The 18-keyword catalog cannot catch novel paraphrased injection syntax without ongoing lexicon maintenance.
- **Inferred Rationale**: [Inferred] Using lightweight base checkpoints (`whisper-base`, `clip-vit-base-patch32`) was chosen to balance inference latency ($\approx 1\text{s}$) with accuracy on CPU and low-tier cloud instances.
