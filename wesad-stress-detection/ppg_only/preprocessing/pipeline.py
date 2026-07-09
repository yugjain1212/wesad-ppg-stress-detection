import numpy as np
from scipy import signal

def preprocess_bvp(raw_signal, fs=64):
    """
    Preprocess the raw BVP signal.
    1. Handle artifacts: linearly interpolate constant segments (>0.5s).
    2. Band-pass filter: 3rd order Butterworth, 0.5-8 Hz.
    
    Args:
        raw_signal (np.ndarray): 1D array of raw BVP data.
        fs (int): Sampling frequency in Hz.
        
    Returns:
        np.ndarray: Filtered BVP signal.
    """
    signal_len = len(raw_signal)
    
    if signal_len == 0:
        return np.array([])
        
    processed_signal = raw_signal.copy()
    
    # 1. Artifact handling (interpolate over flatlines > 0.5s)
    runs = []
    current_run_start = 0
    for i in range(1, signal_len):
        if processed_signal[i] != processed_signal[i-1]:
            if i - current_run_start >= 0.5 * fs:
                runs.append((current_run_start, i))
            current_run_start = i
    if signal_len - current_run_start >= 0.5 * fs:
        runs.append((current_run_start, signal_len))
        
    for start, end in runs:
        # Linearly interpolate between start-1 and end
        if start > 0 and end < signal_len:
            val_start = processed_signal[start-1]
            val_end = processed_signal[end]
            processed_signal[start:end] = np.linspace(val_start, val_end, end-start, endpoint=False)
        elif start == 0 and end < signal_len:
            processed_signal[start:end] = processed_signal[end]
        elif start > 0 and end == signal_len:
            processed_signal[start:end] = processed_signal[start-1]
            
    # 2. Band-pass filter
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 8.0 / nyq
    b, a = signal.butter(3, [low, high], btype='band')
    if len(processed_signal) > 9:
        filtered_signal = signal.filtfilt(b, a, processed_signal)
    else:
        filtered_signal = processed_signal
        
    return filtered_signal

def normalize_window(window_signal):
    """
    Z-score normalization for a window.
    
    Args:
        window_signal (np.ndarray): 1D array of signal window.
        
    Returns:
        np.ndarray: Normalized signal window.
    """
    if len(window_signal) == 0:
        return window_signal
        
    mean_val = np.mean(window_signal)
    std_val = np.std(window_signal)
    if std_val > 1e-8:
        return (window_signal - mean_val) / std_val
    return window_signal - mean_val

def extract_peaks(normalized_signal, fs=64):
    """
    Extract peak indices and amplitudes from a normalized signal window.
    Uses adaptive thresholding and distance constraint.
    
    Args:
        normalized_signal (np.ndarray): 1D array of normalized BVP data.
        fs (int): Sampling frequency in Hz.
        
    Returns:
        tuple: (peak_indices, peak_amplitudes)
    """
    if len(normalized_signal) == 0:
        return np.array([]), np.array([])
        
    mean_val = np.mean(normalized_signal)
    std_val = np.std(normalized_signal)
    
    distance = int(0.33 * fs)
    height_thresh = mean_val + 0.3 * std_val
    
    peaks, _ = signal.find_peaks(
        normalized_signal, 
        distance=max(1, distance), 
        height=height_thresh
    )
    
    amplitudes = normalized_signal[peaks]
    return peaks, amplitudes
