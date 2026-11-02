# DeepShield — image forensic research dashboard

DeepShield is a Streamlit UI for **whole-image AI origin classification** (with
optional independently provided model weights), exploratory forensic signals,
optional TruFor tampering localization and SHA-256/EXIF/C2PA inspection.

**This repository does not include AI model weights or the TruFor repository.**
Installing only `requirements.txt` runs the basic UI and forensic tools; it
does not supply an AI detector. Optional AI packages can be large downloads.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Windows: `powershell -NoProfile -ExecutionPolicy Bypass -File .\KUR_WINDOWS.ps1`
and double-click `BASLAT_WINDOWS.bat`. For optional heavyweight AI packages,
run `KUR_WINDOWS.ps1 -AI` and approve the download.

Bring your own compatible checkpoint at `outputs/deepshield_ai_b0.pt`; source
format described in `training/train_classifier.py`. TruFor must be separately
installed under `models/TruFor/`, subject to its separate **non-commercial
research/informational** licensing conditions. See `THIRD_PARTY_NOTICES.md`.

Scores are not calibrated real-world probabilities, pixel percentages, or
forensic proof. Confidence/reliability requires independent hold-out evaluation.
For data privacy, public deployment, limitations and contribution details, see
`PRIVACY.md`, `SECURITY.md`, `README.md` (Turkish), `CONTRIBUTING.md`.

Tests: `python -m pip install -r requirements-dev.txt && python -m pytest -q tests`.
