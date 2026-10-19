# ADR-0002: Dual-Feature Ensemble (Log-Mel + LFCC) with Gated Calibration for Audio Spoof Detection

> In this page: Architecture Decision Record for combining Log-Mel CNN and LFCC LCNN models with a disagreement-gated wav2vec2 calibrator.  
> Status: Accepted  
> Deciders: Guardian AI Audio Team  
> Date: 2026-04-10

---

## Context
Synthetic speech artifacts vary widely depending on the generation technique:
- Neural Text-to-Speech (TTS) models (e.g. Tacotron, FastSpeech) exhibit phase inconsistencies and spectral envelope smoothing.
- Voice Conversion (VC) models introduce spectral tilt and unnatural pitch transitions.
- Vocoders (e.g. WaveNet, HiFi-GAN) often leave distinct high-frequency spectral signatures that human ears attenuate.

Standard Mel spectrograms compress high frequencies into wider filter bands based on human psychoacoustics, potentially discarding the very high-frequency artifacts left by neural vocoders. Conversely, Linear Frequency Cepstral Coefficients (LFCC) treat all frequency bands linearly, preserving crucial high-frequency forensic evidence.

## Decision
We implemented a dual-stream architecture combining:
1. **Log-Mel Feature Stream**: 80 Mel filters processed by a custom 3-stage CNN with Squeeze-and-Excitation (SE) channel attention.
2. **LFCC Feature Stream**: 60 LFCC coefficients processed by an 8-layer Light CNN (LCNN) utilizing Max-Feature-Map (MFM) activations.
3. **Primary Scratch Ensemble**: A weighted combination:
   $$S_{base} = 0.55 \cdot P_{CNN} + 0.45 \cdot P_{LCNN}$$
4. **Secondary Calibrator Guard**: An optional pretrained `wav2vec2-large-xlsr-deepfake-audio-classification` transformer that is only allowed to pull the score when there is substantial disagreement ($\Delta \ge 0.35$ or $\Delta \ge 0.70$).

## Alternatives Considered
- **Pretrained Transformer Only (wav2vec2 / Whisper)**:
  - *Pros*: High zero-shot transfer on standard benchmarks.
  - *Cons*: High latency (~300–800ms vs ~30ms for CNN/LCNN), heavy memory footprint, fragile dependency on external Hugging Face downloads.
- **Log-Mel Spectrogram Only**:
  - *Pros*: Simpler pipeline, common in speech recognition.
  - *Cons*: Poor sensitivity to high-frequency vocoder ringing and synthetic artifacts.
- **Raw Waveform End-to-End (e.g., RawNet2)**:
  - *Pros*: Eliminates hand-crafted spectrogram extraction.
  - *Cons*: Significantly harder to train stably on smaller voice datasets, higher compute requirements.

## Consequences & Trade-offs
### Positive Consequences:
- **Ultra-Fast Real-Time Inference**: The primary CNN + LCNN pipeline executes in $\sim 30\text{ ms}$ on standard CPU hardware.
- **High Forensic Complementarity**: Log-Mel captures perceptual prosody and phoneme transitions; LFCC captures high-frequency vocoder anomalies.
- **Overconfidence Mitigation**: The gated calibration and extreme score softening prevent the model from collapsing to false $0.0$ or $1.0$ certainty.

### Negative Consequences / Trade-offs:
- **Feature Extraction Overhead**: Must compute both `librosa.feature.melspectrogram` and `torchaudio.transforms.LFCC` for every audio chunk.
- **Dual Model Storage**: Requires shipping two distinct PyTorch `.pth` checkpoint weights (`cnn_best_clean.pth` and `lcnn_best_clean.pth`).
- **Inferred Rationale**: [Inferred] The $0.55 / 0.45$ ensemble weighting was empirically chosen based on validation trials on the Deep Voice Kaggle dataset to slightly favor the SE-CNN model.
