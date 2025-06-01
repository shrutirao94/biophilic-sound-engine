import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import hilbert

input_dir = "data/processed"
output_file = "features/extracted-features.csv"
os.makedirs("features", exist_ok=True)

feature_data = []

for fname in os.listdir(input_dir):
    if fname.endswith(".wav"):
        filepath = os.path.join(input_dir, fname)
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Extract features
        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        # Amplitude envelope (mean of Hilbert envelope)
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal) 
        envelope = np.mean(amplitude_envelope)

        # Onset features
        onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_density = len(onsets) / (len(y) / sr)
        ioi = np.diff(onsets).mean() if len(onsets) > 1 else 0
        
        # Save as a dict
        feature_row = {
            "filename": fname,
            "rms": rms,
            "centroid": centroid,
            "flatness": flatness,
            "bandwidth": bandwidth,
            "zcr": zcr,
            "envelope": envelope,
            "onset_density": onset_density,
            "ioi": ioi
            # Add other features here as needed!
        }
        feature_data.append(feature_row)
        print("Extracted:", fname)

# Save to CSV
df = pd.DataFrame(feature_data)
df.to_csv(output_file, index=False)
print("Features saved to:", output_file)

