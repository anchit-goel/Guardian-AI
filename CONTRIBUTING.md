# Contributing to Guardian AI

Thank you for your interest in contributing to Guardian AI! We welcome bug fixes, documentation improvements, new forensic feature extractors, and performance optimizations.

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior following the guidelines in that document.

---

## Development Environment Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Guardian-AI.git
   cd Guardian-AI
   ```

2. **Create a Virtual Environment (Python 3.10.x required)**:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

---

## Running Verification Tests

Before submitting changes, ensure that all tests pass:

```bash
# 1. Run import sanity test
python scripts/import_test.py

# 2. Run multimodal detector functional test
cd modules/multimodal_injection
python test.py
cd ../..

# 3. Run audio test runner
cd modules/audio_spoof_system
python test_runner.py
cd ../..

# 4. Start local server and test API health
python app.py &
SERVER_PID=$!
sleep 3
curl -f http://127.0.0.1:8080/model-status
kill $SERVER_PID
```

---

## Contribution Guidelines

### Adding a New Model or Modality
Please consult the [Adding a New Modality Guide](docs/guides/adding-a-new-modality-or-model.md) before submitting an architectural PR. Key requirements:
- Isolate feature extraction and inference logic in a dedicated subfolder under `modules/`.
- Ensure new model weights are documented with a formal [Model Card](docs/models/model-cards.md).
- Avoid committing large binary weight files directly to git (> 50 MB); propose hosting weights via Git LFS or Hugging Face Hub.

### Code Style & Formatting *(Proposed)*
- Python code should conform to **PEP 8** style guidelines.
- Keep function signatures type-annotated where practical.
- Preserve existing docstrings and comments.
- Ensure all uploaded temporary files are safely cleaned up inside `try...finally` blocks.

---

## Pull Request Checklist

Before opening a pull request, verify that:
- [ ] Your code runs with Python 3.10.13.
- [ ] No hardcoded absolute personal paths (e.g. `/Users/<name>/`) remain.
- [ ] You have run `scripts/import_test.py` and local smoke tests.
- [ ] New endpoints are documented in `docs/api/reference.md`.
- [ ] Relevant documentation in `/docs` has been updated or created.
- [ ] Commit messages follow standard imperative conventions (e.g. `feat: add temporal video forensic module`).
