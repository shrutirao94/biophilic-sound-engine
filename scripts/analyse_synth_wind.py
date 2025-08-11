import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# === PATHS ===
synth_dir = Path("/Users/shrutirao/Documents/projects/study-3/biophilic-sound-engine/data/processed_classes_filtered/wind/synthetic")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=48000, mono=True)
    harmonic, _ = librosa.effects.hpss(y)
    noise = y - harmonic

    hnr = 10 * np.log10((np.sum(harmonic**2) + 1e-8) / (np.sum(noise**2) + 1e-8))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    return hnr, flatness, centroid, zcr

# === PROCESS ===
data = []
for wav_file in synth_dir.glob("*.wav"):
    hnr, flatness, centroid, zcr = extract_features(wav_file)
    data.append({"file": wav_file.name, "hnr": hnr, "flatness": flatness, "centroid": centroid, "zcr": zcr})

df = pd.DataFrame(data)

# === SUMMARY ===
print("\n=== Synthetic Wind Feature Summary ===")
print(df.describe())

# === PLOTS ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].hist(df["hnr"], bins=30, color="skyblue"); axes[0,0].set_title("HNR (dB)")
axes[0,1].hist(df["flatness"], bins=30, color="salmon"); axes[0,1].set_title("Spectral Flatness")
axes[1,0].hist(df["centroid"], bins=30, color="green"); axes[1,0].set_title("Spectral Centroid (Hz)")
axes[1,1].hist(df["zcr"], bins=30, color="purple"); axes[1,1].set_title("Zero Crossing Rate")
plt.tight_layout()
plt.show()

