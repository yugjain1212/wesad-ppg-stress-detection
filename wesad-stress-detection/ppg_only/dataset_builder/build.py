import os
import pickle
import numpy as np
import pandas as pd
import sys

# Ensure imports work from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ppg_only.preprocessing.pipeline import preprocess_bvp, normalize_window, extract_peaks
from ppg_only.feature_extraction.features import extract_all_features, get_feature_names

# Label mapping
# 0 = Not defined / transient
# 1 = Baseline -> 0
# 2 = Stress -> 1
# 3 = Amusement -> 0
# 4, 5, 6, 7 = Ignore
LABEL_MAPPING = {
    1: 0,
    2: 1,
    3: 0
}

def sync_labels(label_array, bvp_len, label_fs=700, bvp_fs=64):
    """
    Synchronize the 700 Hz labels to 64 Hz BVP signal explicitly via timestamps.
    Returns a list of (start_idx_64, end_idx_64, mapped_label) for valid activity runs.
    """
    runs = []
    
    current_label = label_array[0]
    current_start = 0
    
    for i in range(1, len(label_array)):
        if label_array[i] != current_label:
            # End of run
            end = i
            
            # Map the label
            mapped_label = LABEL_MAPPING.get(int(current_label), None)
            
            if mapped_label is not None:
                # Convert boundaries to timestamp in seconds
                t_start = current_start / label_fs
                t_end = end / label_fs
                
                # Convert timestamps to BVP sample index
                idx_64hz_start = int(round(t_start * bvp_fs))
                idx_64hz_end = int(round(t_end * bvp_fs))
                
                # Ensure it doesn't exceed BVP array length
                idx_64hz_start = min(idx_64hz_start, bvp_len)
                idx_64hz_end = min(idx_64hz_end, bvp_len)
                
                if idx_64hz_start < idx_64hz_end:
                    runs.append((idx_64hz_start, idx_64hz_end, mapped_label))
                    
            current_label = label_array[i]
            current_start = i
            
    # Handle last run
    mapped_label = LABEL_MAPPING.get(int(current_label), None)
    if mapped_label is not None:
        t_start = current_start / label_fs
        t_end = len(label_array) / label_fs
        idx_64hz_start = int(round(t_start * bvp_fs))
        idx_64hz_end = int(round(t_end * bvp_fs))
        idx_64hz_start = min(idx_64hz_start, bvp_len)
        idx_64hz_end = min(idx_64hz_end, bvp_len)
        if idx_64hz_start < idx_64hz_end:
            runs.append((idx_64hz_start, idx_64hz_end, mapped_label))
            
    return runs

def window_and_extract(filtered_bvp, valid_runs, subject_id, window_len_sec=60, stride_sec=30, fs=64):
    """
    Window the continuous filtered BVP signal, enforce label purity, and extract features.
    """
    window_samples = window_len_sec * fs
    stride_samples = stride_sec * fs
    
    features_list = []
    
    for start_idx, end_idx, label in valid_runs:
        current_idx = start_idx
        
        while current_idx + window_samples <= end_idx:
            # We have a valid pure window
            window_signal = filtered_bvp[current_idx : current_idx + window_samples]
            
            # Normalize window
            norm_window = normalize_window(window_signal)
            
            # Extract peaks
            peaks, amps = extract_peaks(norm_window, fs=fs)
            
            # Extract all features
            feats = extract_all_features(norm_window, peaks, amps, fs=fs)
            
            # Add label and subject ID
            row = list(feats) + [label, subject_id]
            features_list.append(row)
            
            current_idx += stride_samples
            
    return features_list

def process_subject(subject_id, dataset_dir):
    """Process a single subject's WESAD data."""
    print(f"Processing subject {subject_id}...")
    pkl_path = os.path.join(dataset_dir, subject_id, f"{subject_id}.pkl")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    bvp_signal = data['signal']['wrist']['BVP'].flatten()
    label_array = data['label'].flatten()
    
    # 1. Filter the whole continuous per-subject signal first
    filtered_bvp = preprocess_bvp(bvp_signal, fs=64)
    
    # 2. Synchronize labels
    valid_runs = sync_labels(label_array, len(bvp_signal), label_fs=700, bvp_fs=64)
    
    # 3. Window and extract features
    features_list = window_and_extract(filtered_bvp, valid_runs, subject_id)
    
    return features_list

def build_dataset(dataset_dir, output_dir):
    """Build the complete dataset from all subjects."""
    subjects = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    
    all_features = []
    
    for subj in subjects:
        try:
            feats = process_subject(subj, dataset_dir)
            all_features.extend(feats)
        except Exception as e:
            print(f"Error processing {subj}: {e}")
            
    # Create DataFrame
    feature_names = get_feature_names()
    columns = feature_names + ['label', 'subject_id']
    
    df = pd.DataFrame(all_features, columns=columns)
    
    # Fill NaNs with 0 instead of dropping to retain windows
    df = df.fillna(0)
    
    # Check class imbalance
    val_counts = df['label'].value_counts()
    print("Class distribution:")
    print(val_counts)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset saved to {out_path} with shape {df.shape}")
    
if __name__ == "__main__":
    dataset_dir = "/Volumes/exssd/internshipmodel/dataset/WESAD"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    build_dataset(dataset_dir, output_dir)
