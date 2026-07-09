import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppg_only.feature_extraction.features import extract_all_features, get_feature_names
from ppg_only.preprocessing.pipeline import extract_peaks

class TestFeatures(unittest.TestCase):
    def test_feature_extraction(self):
        """Test feature extraction output shape."""
        fs = 64
        # Sine wave with ~1 Hz frequency (60 bpm)
        t = np.linspace(0, 60, 60 * fs)
        signal = np.sin(2 * np.pi * 1.0 * t)
        
        # Add some noise
        np.random.seed(42)
        signal += np.random.normal(0, 0.1, len(signal))
        
        peaks, amps = extract_peaks(signal, fs=fs)
        
        feats = extract_all_features(signal, peaks, amps, fs=fs)
        names = get_feature_names()
        
        self.assertEqual(len(feats), len(names))
        # Ensure no nan values for a clean signal with peaks
        self.assertFalse(np.isnan(feats).any())

if __name__ == '__main__':
    unittest.main()
