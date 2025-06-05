import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import hilbert

# Input directory for processed nature segments
input_dir = "data/processed/nature/"
output_file = "features/nature/extracted-features.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

feature_data = []

# Loop through each nature subfolder
for nature_source in os.listdir(input_dir):
    source_dir = os.path.join(input_dir, nature_source)
    if os.path.isdir(source_dir):
        for fname in os.listdir(source_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(source_dir, fname)
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

                # Save feature row
                feature_row = {
                    "source": nature_source,
                    "filename": fname,
                    "rms": rms,
                    "centroid": centroid,
                    "flatness": flatness,
                    "bandwidth": bandwidth,
                    "zcr": zcr,
                    "envelope": envelope,
                    "onset_density": onset_density,
                    "ioi": ioi
                }
                feature_data.append(feature_row)
                print("Extracted:", nature_source, fname)

# Save to CSV
df = pd.DataFrame(feature_data)
df.to_csv(output_file, index=False)
print("✅ Features for nature segments saved to:", output_file)

