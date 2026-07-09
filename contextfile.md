# Codex Build Spec — WESAD PPG-Only Stress Detection (Raspberry Pi 4B Deployment)

## 0. Context (read this first)

This is the **deployment branch** of an existing AIoT mental health monitoring
project. A separate, already-built **multimodal** model (ECG + EDA + EMG +
Resp + Temp + ACC, chest+wrist WESAD data) exists for the research
submission. This PPG-only model is a **second, independent model** trained
so it can run live on:

- Raspberry Pi 4B (aarch64, Debian-based Raspberry Pi OS)
- MAX30102 PPG sensor over I2C

The training data modality and the deployment sensor modality **must match
exactly** — Wrist BVP (PPG) only. Do not use any other WESAD signal for
training. Do not use chest data at all in this branch.

Repository layout (this branch lives alongside the multimodal one):

```
wesad-stress-detection/
├── multimodal/              ← existing model, do not touch
├── ppg_only/                ← THIS PROJECT
│   ├── data/
│   ├── preprocessing/
│   ├── windowing/
│   ├── feature_extraction/
│   ├── dataset_builder/
│   ├── models/
│   ├── evaluation/
│   ├── saved_models/
│   ├── results/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
└── raspberry_pi/             ← live inference deployment code
    ├── sensor_reader.py
    ├── live_predict.py
    ├── config.yaml
    └── requirements_pi.txt
```

**Non-negotiable design rule:** the feature-extraction function used in
`ppg_only/feature_extraction/` and the one used in
`raspberry_pi/live_predict.py` must be the **same function**, imported from a
single shared module (not copy-pasted). This is the single most important
correctness requirement — any drift here silently invalidates the deployed
model's predictions relative to what it was trained on.

---

## 1. Objective

Binary stress classification (stress vs. non-stress) using **only** the
WESAD wrist BVP (PPG) signal, trained offline, then deployed for live
inference on a Raspberry Pi 4B reading a MAX30102 sensor over I2C.

---

## 2. Dataset

**Source:** WESAD (Wearable Stress and Affect Detection), pickle files per
subject.

**Signal used (only this one):**
```python
data["signal"]["wrist"]["BVP"]   # sampled at 64 Hz
```

**Explicitly excluded:** chest ECG, EDA, EMG, Resp, Temp, chest ACC, wrist
ACC, wrist EDA, wrist TEMP.

**Subjects used (15 total):**
```
S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17
```
**Excluded:** S1, S12 (per WESAD documentation — incomplete/invalid data).

---

## 3. Labels

**Raw label array:** `data["label"]`, sampled at **700 Hz**.

**WESAD label codes:**
```
0 = Not defined / transient
1 = Baseline
2 = Stress
3 = Amusement
4 = Meditation
5, 6, 7 = Ignore (recovery/other protocol markers)
```

**Binary mapping used for this task:**
```
Baseline (1)   → 0  (non-stress)
Amusement (3)  → 0  (non-stress)
Stress (2)     → 1  (stress)
```

**Discard entirely:** Meditation (4), transient/undefined (0), and any other
code (5,6,7). Do not fold these into either class.

---

## 4. Critical: Label/Signal Synchronization

The label array and the BVP array run at **different sample rates** and
**must not be indexed as if aligned 1:1**.

- Label rate: 700 Hz
- Wrist BVP rate: 64 Hz
- Sample rate ratio: `700 / 64 = 10.9375`

**Correct procedure:**
1. Scan `data["label"]` (700 Hz index space) to find contiguous runs of a
   single label value → gives `(start_idx_700hz, end_idx_700hz, label)` per
   run.
2. Convert each boundary from 700 Hz sample index to a **timestamp in
   seconds**: `t = idx_700hz / 700.0`.
3. Convert that timestamp to a BVP sample index:
   `idx_64hz = round(t * 64.0)`.
4. Slice `BVP[idx_64hz_start : idx_64hz_end]` for that activity run.

Do **not** do `idx_64hz = idx_700hz` and do not do
`idx_64hz = idx_700hz * (64/700)` applied directly to per-sample label
arrays without going through the timestamp conversion above — implement it
via timestamps as specified, since off-by-ratio slicing errors are the most
common source of silent mislabeling in WESAD pipelines.

Write a unit test that checks: for a synthetic label array of known runs,
the converted BVP segment boundaries land within ±1 sample of hand-computed
expected values.

---

## 5. Windowing

- **Window length:** 60 seconds → `60 * 64 = 3840` samples per window.
- **Stride:** 30 seconds → `30 * 64 = 1920` samples (50% overlap).
- **Label purity rule:** a window is kept only if every sample in it falls
  within a single continuous same-label activity run (from Section 4). If a
  window would straddle two different activity runs (e.g. end of Baseline
  into start of Stress), **discard it**. Do not majority-vote mixed windows.
- Windows shorter than 3840 samples at the end of a run (leftover tail) are
  discarded, not zero-padded.

---

## 6. Signal Processing Pipeline (per subject, before windowing)

Apply in this order to the raw per-subject BVP signal:

1. **Band-pass filter:** Butterworth, order 3–4, passband **0.5–8 Hz**
   (covers HR range ~30–480 bpm with margin; PPG pulse fundamental is
   typically 0.7–3 Hz). Use `scipy.signal.butter` + `filtfilt` (zero-phase,
   important since peak timing matters for HRV).
2. **Noise/artifact handling:** clip or interpolate over flat-line /
   saturation segments (MAX30102 finger-off or motion artifact simulation).
   For WESAD offline data, detect segments where the raw signal is
   constant for >0.5s (sensor saturation) and linearly interpolate.
3. **Normalization:** z-score normalize per-window (not per-subject) so the
   deployment pipeline can normalize a live rolling window the same way
   without needing subject-level statistics that don't exist in real time.
4. **Peak detection:** `scipy.signal.find_peaks` on the filtered,
   normalized signal with:
   - `distance = int(0.33 * fs)` (refractory ~330ms → caps HR at ~180bpm)
   - `height` = adaptive threshold, e.g. `mean + 0.3 * std` of the window
   - Record peak indices and peak amplitudes per window; these feed both
     HR/HRV and pulse-wave features below.
5. Windowing happens **after** filtering (filter the whole continuous
   per-subject signal first, then window — avoids edge artifacts from
   filtering short 60s segments independently).

This exact 5-step function must live in
`ppg_only/preprocessing/pipeline.py` as a single callable,
e.g. `preprocess_bvp(raw_signal, fs=64) -> filtered_signal`, and
`extract_peaks(filtered_signal, fs=64) -> peak_indices, peak_amplitudes`,
imported by both the offline training code and
`raspberry_pi/live_predict.py`.

---

## 7. Feature Extraction (per 60s window)

All features must be computable in real time from a rolling 60s buffer of
MAX30102 output — do not include any feature that needs information outside
the window (e.g. no global/subject-level normalization).

### Time domain (on filtered, normalized signal)
- Mean, Median, Standard deviation, Variance, RMS, Skewness, Kurtosis,
  Signal energy (`sum(x^2)`)

### Heart rate features (derived from peak-to-peak intervals)
- Mean HR, Max HR, Min HR, HR standard deviation
  (HR per beat = `60 / RR_interval_seconds`)

### HRV features (time domain, from RR intervals in ms)
- Mean RR interval
- SDNN (std dev of RR intervals)
- RMSSD (root mean square of successive RR differences)
- SDSD (std dev of successive RR differences)
- pNN50 (% of successive RR diffs > 50ms)

### Pulse wave morphology features
- Peak count (in window)
- Mean peak-to-peak interval
- Mean peak amplitude
- Pulse width (at half-max amplitude, per pulse, averaged)
- Rise time (foot-to-peak, averaged)
- Fall time (peak-to-foot, averaged)

### Frequency domain (Welch PSD on filtered window)
- LF power (0.04–0.15 Hz)
- HF power (0.15–0.4 Hz)
- LF/HF ratio
- Spectral entropy

> **Caveat to keep, not hide:** 60-second windows are short for
> frequency-domain HRV (LF/HF conventionally wants 2–5 minute windows for
> stable estimates). Keep these features since the assignment/deployment
> constraints require 60s windows, but log this as a stated methodological
> limitation in `README.md` and in any report — do not present LF/HF from
> 60s windows as clinically standard-length HRV without the caveat.

**Output:** one row per window, feature columns + `label` (0/1) +
`subject_id` (kept for LOSO grouping, dropped before model input).

---

## 8. Machine Learning

**Models to train and compare:**
- Logistic Regression
- Random Forest
- SVM (RBF kernel)

**Preprocessing for ML:**
- `StandardScaler` fit **only on training folds**, applied to train and
  test folds separately (no leakage).
- No SMOTE specified for this PPG-only branch unless class imbalance in the
  binary labels is severe (check WESAD Stress vs. Baseline+Amusement ratio
  first, then decide — log the ratio either way).

**Persist artifacts:**
```
saved_models/scaler.pkl
saved_models/model.pkl          # best model by evaluation criteria below
saved_models/model_metadata.json  # feature order, model type, training date, LOSO summary metrics
```
`model_metadata.json` must include the exact ordered list of feature names
the model expects — `raspberry_pi/live_predict.py` reads this to build the
feature vector in the right order without hardcoding it twice.

---

## 9. Evaluation

**Cross-validation:** Leave-One-Subject-Out (LOSO) across the 15 subjects.

**Metrics per fold and aggregated (mean ± std across folds):**
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

**Artifacts to generate in `results/`:**
- Confusion matrix (aggregated across folds)
- ROC curve (aggregated or per-fold overlay)
- Precision-Recall curve
- A results table (CSV + printed) with per-subject and mean metrics

---

## 10. Raspberry Pi 4B Deployment

### 10.1 Hardware/software stack
- Raspberry Pi 4B, Raspberry Pi OS (64-bit recommended for TensorFlow/sklearn wheel availability)
- MAX30102 over I2C (SDA/SCL, 3.3V logic — confirm sensor breakout board is 3.3V tolerant before wiring to Pi GPIO)
- Python library for MAX30102: use a maintained I2C driver
  (e.g. `smbus2` for raw I2C register access, or an existing MAX30102 Python
  driver package). Do not assume a specific PyPI package name without
  verifying it's still maintained and installable on Raspberry Pi OS at
  build time — check `pip index versions <package>` or PyPI directly rather
  than hardcoding a possibly-abandoned package.

### 10.2 Real-time constraints to design for
- MAX30102 raw sample rate is configurable (commonly 100 Hz); the
  deployment reader must **resample/decimate to 64 Hz** to match training,
  or the feature extraction (peak detection thresholds, filter cutoffs) will
  be wrong for the new sample rate. Pick one approach explicitly and
  document it — do not silently feed 100 Hz data into a 64 Hz-tuned
  pipeline.
- Maintain a rolling buffer of the last 60 seconds of samples; run
  inference every 30 seconds (matching training stride) on the current
  buffer, not sample-by-sample.
- `raspberry_pi/live_predict.py` must import `preprocess_bvp`,
  `extract_peaks`, and the feature extraction function from the shared
  `ppg_only/` modules — not reimplement them.
- Load `scaler.pkl`, `model.pkl`, and `model_metadata.json` at startup;
  build the feature vector in the exact column order from the metadata file
  before calling `scaler.transform()`.

### 10.3 Failure modes to handle explicitly
- Finger-off / low perfusion index from MAX30102 → detect via saturation/flatline check (Section 6.2 logic reused) and skip prediction with a clear "no signal" state rather than emitting a stress/non-stress label on garbage input.
- I2C read errors/timeouts → retry with backoff, don't crash the loop.

---

## 11. Code Requirements

- Modular, one responsibility per file/module.
- Docstrings on every public function stating units (Hz, seconds, ms) —
  unit mismatches (samples vs. seconds vs. ms) are the most likely class of
  bug in this pipeline.
- Reproducible: fixed random seeds for model training and any stochastic
  step.
- The shared preprocessing/feature-extraction module must have zero
  dependency on WESAD-specific data structures, so it can be imported
  unmodified by `raspberry_pi/live_predict.py`.
- `requirements.txt` (training, x86/dev machine) and
  `requirements_pi.txt` (deployment, must be pip-installable on Raspberry
  Pi OS aarch64 — avoid packages without arm64 wheels; note if any dependency
  needs building from source on the Pi and roughly how long that takes).

---

## 12. Deliverables Checklist

- [ ] `ppg_only/` full pipeline: load → sync → window → preprocess → features → train → evaluate
- [ ] `train.py`, `evaluate.py`, `predict.py` (offline batch predict, for sanity-checking against `raspberry_pi/live_predict.py` output on the same data)
- [ ] `saved_models/{scaler.pkl, model.pkl, model_metadata.json}`
- [ ] `results/` with LOSO metrics table + confusion matrix + ROC + PR curves
- [ ] `raspberry_pi/sensor_reader.py`, `live_predict.py`, `config.yaml`, `requirements_pi.txt`
- [ ] `README.md` documenting: sync method, windowing rule, feature list, LOSO results summary, the 60s-window LF/HF caveat, and Pi wiring/sample-rate notes
