import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppg_only.dataset_builder.build import sync_labels

class TestSyncLabels(unittest.TestCase):
    def test_sync_labels(self):
        """
        Test that for a synthetic label array of known runs, the converted
        BVP segment boundaries land within +/- 1 sample of hand-computed values.
        
        Let's say:
        Label fs = 700 Hz
        BVP fs = 64 Hz
        
        Run 1: 0 to 1400 samples (2 seconds) -> Label 1 (Baseline -> 0)
        Run 2: 1400 to 2800 samples (2 seconds) -> Label 2 (Stress -> 1)
        Run 3: 2800 to 4200 samples (2 seconds) -> Label 3 (Amusement -> 0)
        
        Total label length = 4200.
        Total BVP length = 6 seconds * 64 Hz = 384 samples.
        
        Hand-computed BVP boundaries:
        Run 1: t=0s to t=2s -> idx 0 to 128
        Run 2: t=2s to t=4s -> idx 128 to 256
        Run 3: t=4s to t=6s -> idx 256 to 384
        """
        label_array = np.concatenate([
            np.ones(1400),       # Label 1
            np.ones(1400) * 2,   # Label 2
            np.ones(1400) * 3    # Label 3
        ])
        
        bvp_len = 384
        
        runs = sync_labels(label_array, bvp_len, label_fs=700, bvp_fs=64)
        
        self.assertEqual(len(runs), 3)
        
        # Run 1: expected (0, 128, 0)
        self.assertTrue(abs(runs[0][0] - 0) <= 1)
        self.assertTrue(abs(runs[0][1] - 128) <= 1)
        self.assertEqual(runs[0][2], 0)
        
        # Run 2: expected (128, 256, 1)
        self.assertTrue(abs(runs[1][0] - 128) <= 1)
        self.assertTrue(abs(runs[1][1] - 256) <= 1)
        self.assertEqual(runs[1][2], 1)
        
        # Run 3: expected (256, 384, 0)
        self.assertTrue(abs(runs[2][0] - 256) <= 1)
        self.assertTrue(abs(runs[2][1] - 384) <= 1)
        self.assertEqual(runs[2][2], 0)

if __name__ == '__main__':
    unittest.main()
