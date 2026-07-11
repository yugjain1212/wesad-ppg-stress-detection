# WESAD PPG-Only Stress Detection

This repository contains the offline training pipeline and the Raspberry Pi deployment code for a binary stress classification model. 
The model predicts Stress vs. Non-Stress using **only** the wrist BVP (PPG) signal from the WESAD dataset.

## Methodological Limitation: 60s Window HRV
> **IMPORTANT NOTE**: 
> This pipeline calculates frequency-domain HRV features (LF power, HF power, LF/HF ratio) from 60-second windows. While this meets the deployment latency constraint (running every 30s on a 60s rolling buffer), clinical standard LF/HF estimation conventionally requires 2-5 minute windows. Be aware that these frequency-domain features may have higher variance or differ from standard clinical metrics.

## Synchronization and Windowing
- **Synchronization**: The dataset's 700 Hz labels are explicitly synchronized to the 64 Hz BVP signal using calculated timestamp boundaries.
- **Windowing Rule**: We extract 60-second windows with a 30-second stride. A window is only accepted if the entire 60s segment belongs to a single, continuous label run. Mixed or straddling windows are discarded.

## Extracted Features
The following features are extracted per window:
1. **Time Domain**: Mean, Median, Standard deviation, Variance, RMS, Skewness, Kurtosis, Signal energy
2. **Heart Rate (HR)**: Mean HR, Max HR, Min HR, HR standard deviation
3. **HRV (Time Domain)**: Mean RR interval, SDNN, RMSSD, SDSD, pNN50
4. **Pulse Wave Morphology**: Peak count, Mean peak-to-peak interval, Mean peak amplitude, Mean pulse width, Mean rise time, Mean fall time
5. **Frequency Domain**: LF power, HF power, LF/HF ratio, Spectral entropy

## Training Pipeline
1. **Build Dataset**: `python ppg_only/dataset_builder/build.py`
2. **Evaluate (LOSO CV)**: `python ppg_only/evaluate.py` (Generates LOSO metrics and plots in `results/`)
3. **Train Final Model**: `python ppg_only/train.py` (Saves model, scaler, and metadata to `saved_models/`)
4. **Predict (Offline)**: `python ppg_only/predict.py` (Runs predictions on the offline dataset for sanity checking)

## Raspberry Pi 4B Deployment
The deployment code runs on a Raspberry Pi 4B connected to a MAX30102 sensor via I2C (3.3V logic).

### Setup
1. Ensure I2C is enabled on the Raspberry Pi (`sudo raspi-config`).
2. Install dependencies: `pip install -r raspberry_pi/requirements_pi.txt`. Some packages (like `scipy` or `scikit-learn`) may take time to build or require binary wheels for `aarch64`.
3. Check `raspberry_pi/config.yaml` to configure target frequencies and buffer sizes.

### Usage
Run the live predictor:
```bash
python raspberry_pi/live_predict.py
```
This script reads samples continuously, resamples them from 100 Hz to 64 Hz, maintains a 60-second rolling buffer, and performs inference every 30 seconds.
