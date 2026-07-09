import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis, entropy

def extract_time_domain_features(window_signal):
    """Extract time domain features from a normalized BVP window."""
    if len(window_signal) == 0:
        return [np.nan] * 8
        
    mean = np.mean(window_signal)
    median = np.median(window_signal)
    std = np.std(window_signal)
    var = np.var(window_signal)
    rms = np.sqrt(np.mean(window_signal**2))
    skewness = skew(window_signal)
    kurt = kurtosis(window_signal)
    energy = np.sum(window_signal**2)
    
    return [mean, median, std, var, rms, skewness, kurt, energy]

def extract_hr_hrv_pulse_features(peak_indices, peak_amplitudes, window_signal, fs=64):
    """
    Extract Heart Rate, HRV, and pulse morphology features.
    Args:
        peak_indices (np.ndarray): Indices of peaks in the window.
        peak_amplitudes (np.ndarray): Amplitudes of the peaks.
        window_signal (np.ndarray): The normalized signal window.
        fs (int): Sampling frequency.
    Returns:
        List of HR, HRV, and pulse morphology features.
    """
    if len(peak_indices) < 2:
        # Not enough peaks to compute interval features
        return [np.nan] * 15 # 4 HR + 5 HRV + 6 Pulse
        
    # RR intervals in seconds and milliseconds
    rr_intervals_s = np.diff(peak_indices) / fs
    rr_intervals_ms = rr_intervals_s * 1000
    
    # HR features
    hr_array = 60.0 / rr_intervals_s
    mean_hr = np.mean(hr_array)
    max_hr = np.max(hr_array)
    min_hr = np.min(hr_array)
    std_hr = np.std(hr_array)
    
    # HRV features (from RR in ms)
    mean_rr = np.mean(rr_intervals_ms)
    sdnn = np.std(rr_intervals_ms)
    rr_diff = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) > 0 else np.nan
    sdsd = np.std(rr_diff) if len(rr_diff) > 0 else np.nan
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100 if len(rr_diff) > 0 else np.nan
    
    # Pulse wave morphology features
    peak_count = len(peak_indices)
    mean_peak_to_peak = np.mean(rr_intervals_s)
    mean_peak_amp = np.mean(peak_amplitudes)
    
    # Calculate rise time, fall time, pulse width (simple approximations)
    rise_times = []
    fall_times = []
    pulse_widths = []
    
    # We will find the "foot" (valley) before and after each peak
    # A simple way: minimum between consecutive peaks
    for i in range(1, len(peak_indices)-1):
        prev_peak = peak_indices[i-1]
        curr_peak = peak_indices[i]
        next_peak = peak_indices[i+1]
        
        valley_before = prev_peak + np.argmin(window_signal[prev_peak:curr_peak])
        valley_after = curr_peak + np.argmin(window_signal[curr_peak:next_peak])
        
        rise_times.append((curr_peak - valley_before) / fs)
        fall_times.append((valley_after - curr_peak) / fs)
        
        # Pulse width at half-max amplitude
        half_max = peak_amplitudes[i] / 2
        # Find where signal crosses half_max on the left and right
        left_cross = curr_peak
        while left_cross > valley_before and window_signal[left_cross] > half_max:
            left_cross -= 1
        right_cross = curr_peak
        while right_cross < valley_after and window_signal[right_cross] > half_max:
            right_cross += 1
            
        pulse_widths.append((right_cross - left_cross) / fs)
        
    mean_rise_time = np.mean(rise_times) if rise_times else np.nan
    mean_fall_time = np.mean(fall_times) if fall_times else np.nan
    mean_pulse_width = np.mean(pulse_widths) if pulse_widths else np.nan
    
    return [
        mean_hr, max_hr, min_hr, std_hr,
        mean_rr, sdnn, rmssd, sdsd, pnn50,
        peak_count, mean_peak_to_peak, mean_peak_amp, mean_pulse_width, mean_rise_time, mean_fall_time
    ]

def extract_frequency_features(window_signal, fs=64):
    """
    Extract frequency domain features using Welch's PSD.
    Args:
        window_signal (np.ndarray): 1D array of signal window.
        fs (int): Sampling frequency.
    Returns:
        List of frequency domain features.
    """
    if len(window_signal) == 0:
        return [np.nan] * 4
        
    freqs, psd = signal.welch(window_signal, fs=fs, nperseg=min(len(window_signal), 256))
    
    # LF power (0.04-0.15 Hz)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    lf_power = np.trapezoid(psd[lf_mask], freqs[lf_mask]) if hasattr(np, 'trapezoid') else np.trapz(psd[lf_mask], freqs[lf_mask])
    
    # HF power (0.15-0.4 Hz)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask]) if hasattr(np, 'trapezoid') else np.trapz(psd[hf_mask], freqs[hf_mask])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    
    # Spectral entropy
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spec_entropy = entropy(psd_norm)
    
    return [lf_power, hf_power, lf_hf_ratio, spec_entropy]

def get_feature_names():
    """Return a list of feature names in the exact order they are extracted."""
    time_names = [
        "mean", "median", "std", "var", "rms", "skewness", "kurtosis", "energy"
    ]
    hr_hrv_pulse_names = [
        "mean_hr", "max_hr", "min_hr", "std_hr",
        "mean_rr", "sdnn", "rmssd", "sdsd", "pnn50",
        "peak_count", "mean_peak_to_peak", "mean_peak_amp", "mean_pulse_width", "mean_rise_time", "mean_fall_time"
    ]
    freq_names = [
        "lf_power", "hf_power", "lf_hf_ratio", "spectral_entropy"
    ]
    return time_names + hr_hrv_pulse_names + freq_names

def extract_all_features(normalized_signal, peak_indices, peak_amplitudes, fs=64):
    """
    Extract all features from a preprocessed, normalized window.
    Returns a 1D numpy array of features.
    """
    time_feats = extract_time_domain_features(normalized_signal)
    hr_pulse_feats = extract_hr_hrv_pulse_features(peak_indices, peak_amplitudes, normalized_signal, fs=fs)
    freq_feats = extract_frequency_features(normalized_signal, fs=fs)
    
    return np.array(time_feats + hr_pulse_feats + freq_feats)
