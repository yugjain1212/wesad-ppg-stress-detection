import os
import sys
import yaml
import json
import pickle
import time
import numpy as np
import pandas as pd

# Add project root to path to import shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppg_only.preprocessing.pipeline import preprocess_bvp, normalize_window, extract_peaks
from ppg_only.feature_extraction.features import extract_all_features
from sensor_reader import SensorReader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def is_signal_valid(raw_signal, fs):
    """
    Check if the signal has a valid perfusion index / isn't just flatlines.
    A flatline > 0.5s indicates finger-off or sensor error.
    """
    diff = np.diff(raw_signal)
    is_flat = (diff == 0)
    flat_indices = np.where(is_flat)[0]
    
    runs = []
    if len(flat_indices) > 0:
        current_run_start = flat_indices[0]
        for i in range(1, len(flat_indices)):
            if flat_indices[i] != flat_indices[i-1] + 1:
                runs.append(flat_indices[i-1] - current_run_start + 1)
                current_run_start = flat_indices[i]
        runs.append(flat_indices[-1] - current_run_start + 1)
        
    if any(r >= 0.5 * fs for r in runs):
        return False
    return True

def run_live_inference(config):
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ppg_only', 'saved_models'))
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'model.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    if not all(os.path.exists(p) for p in [scaler_path, model_path, metadata_path]):
        print("Model artifacts not found. Please train the model offline first.")
        return
        
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    feature_names = metadata['feature_names']
    fs = config['target_fs']
    window_sec = config['window_sec']
    stride_sec = config['stride_sec']
    
    window_samples = window_sec * fs
    
    reader = SensorReader(target_fs=fs, sensor_fs=config['sensor_fs'])
    
    print(f"Starting live prediction loop. Initializing {window_sec}s buffer...")
    
    # Initialize rolling buffer
    initial_buffer = reader.read_samples(window_sec)
    buffer = list(initial_buffer)
    
    while True:
        # Check signal validity before predicting
        if not is_signal_valid(buffer, fs):
            print("State: NO SIGNAL (Finger off / saturation detected). Skipping prediction.")
        else:
            # Predict
            window_signal = np.array(buffer)
            
            # 1. Preprocess
            filtered_bvp = preprocess_bvp(window_signal, fs=fs)
            
            # 2. Normalize
            norm_bvp = normalize_window(filtered_bvp)
            
            # 3. Peak detection
            peaks, amps = extract_peaks(norm_bvp, fs=fs)
            
            # 4. Feature extraction
            feats = extract_all_features(norm_bvp, peaks, amps, fs=fs)
            
            # Create dataframe to match expected feature order
            df = pd.DataFrame([feats], columns=feature_names)
            
            # 5. Scale and Predict
            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]
            
            state = "STRESS" if prediction == 1 else "NON-STRESS"
            print(f"State: {state} | Stress Probability: {probability:.3f}")
            
        # Shift buffer by stride
        print(f"Waiting {stride_sec}s for next prediction window...")
        new_samples = reader.read_samples(stride_sec)
        
        buffer = buffer[len(new_samples):] + list(new_samples)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_config(config_path)
    try:
        run_live_inference(config)
    except KeyboardInterrupt:
        print("\nLive prediction stopped by user.")
