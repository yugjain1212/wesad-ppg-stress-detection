import time
import numpy as np
from scipy import signal

# Mock or actual import for MAX30102. 
# We'll assume the use of a standard PyPI library like `max30102` which is I2C based.
try:
    from max30102 import MAX30102
except ImportError:
    MAX30102 = None
    print("Warning: max30102 library not found. Will run in mock mode if instantiated.")

class SensorReader:
    def __init__(self, target_fs=64, sensor_fs=100):
        """
        Initialize the MAX30102 sensor reader.
        Args:
            target_fs (int): Target sampling frequency to match training data (64 Hz).
            sensor_fs (int): Raw sampling frequency of the MAX30102 (e.g., 100 Hz).
        """
        self.target_fs = target_fs
        self.sensor_fs = sensor_fs
        
        self.sensor = None
        self._initialize_sensor()
        
    def _initialize_sensor(self):
        if MAX30102 is not None:
            try:
                # Assuming default I2C bus and I2C address
                self.sensor = MAX30102()
                self.sensor.setup_sensor()
                print("MAX30102 initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize MAX30102: {e}")
                self.sensor = None
        else:
            print("Running in mock mode. No real sensor connected.")
            
    def read_samples(self, duration_sec):
        """
        Read samples from the sensor for a given duration.
        Resamples the collected data from sensor_fs to target_fs.
        """
        num_expected_samples = int(duration_sec * self.sensor_fs)
        raw_bvp = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration_sec:
            try:
                if self.sensor:
                    red, ir = self.sensor.read_sequential()
                    # We typically use the IR or RED channel for PPG. WESAD BVP is related to PPG.
                    # We will use IR channel as it usually provides better PPG signal penetration.
                    raw_bvp.append(ir)
                else:
                    # Mock data if no sensor
                    raw_bvp.append(np.sin(2 * np.pi * 1.5 * (time.time() - start_time)) * 1000 + 50000)
                    time.sleep(1.0 / self.sensor_fs)
                    
            except Exception as e:
                print(f"I2C read error: {e}")
                # Handle disconnection or timeout with backoff
                time.sleep(0.1)
                self._initialize_sensor() # Try to reconnect
                
        # If we didn't get enough samples due to errors, pad with the last value or zeros
        if len(raw_bvp) == 0:
            raw_bvp = [0] * num_expected_samples
            
        # Decimate/resample to target_fs (64 Hz)
        # Using scipy.signal.resample to change the sampling rate from 100Hz to 64Hz
        num_target_samples = int(duration_sec * self.target_fs)
        resampled_bvp = signal.resample(raw_bvp, num_target_samples)
        
        return resampled_bvp
