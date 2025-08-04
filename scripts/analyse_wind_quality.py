import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf

# =======================
# CONFIG
# =======================
WIND_DIR = Path("data/processed_classes_filtered/wind")
OUTPUT_CSV = "wind_detailed_quality_stats.csv"
TARGET_SR = 48000

# =======================
# HELPER FUNCTIONS
# =======================

def calculate_silence_ratio(y, threshold=0.01):
    """Proportion of samples considered silent."""
    return np.mean(np.abs(y) < threshold)

def spectral_centroid(y, sr):
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

def spectral_bandwidth(y, sr):
    return float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

def spectral_flatness(y):
    return float(np.mean(librosa.feature.spectral_flatness(y=y)))

def zero_crossing_rate(y):
    return float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

def dynamic_range(y):
    """Difference between 95th and 5th percentile amplitude."""
    return float(np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5))

def crest_factor(y):
    """Peak amplitude to RMS ratio."""
    return float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9))

# =======================
# MAIN ANALYSIS
# =======================
file_stats = []
print("📊 Analyzing wind dataset quality with advanced metrics...")

for wav_file in WIND_DIR.glob("*"):
    if wav_file.suffix.lower() not in [".wav", ".mp3", ".flac"]:
        continue

    try:
        # Load file
        y, sr = librosa.load(wav_file, sr=TARGET_SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if len(y) == 0:
            print(f"⚠️ Skipping empty file: {wav_file.name}")
            continue

        # Compute metrics
        rms = np.sqrt(np.mean(y**2))
        silence_ratio = calculate_silence_ratio(y)
        centroid = spectral_centroid(y, sr)
        bandwidth = spectral_bandwidth(y, sr)
        flatness = spectral_flatness(y)
        zcr = zero_crossing_rate(y)
        dyn_range = dynamic_range(y)
        crest = crest_factor(y)

        file_stats.append({
            "filename": wav_file.name,
            "duration_s": duration,
            "rms": rms,
            "silence_ratio": silence_ratio,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_flatness": flatness,
            "zero_crossing_rate": zcr,
            "dynamic_range": dyn_range,
            "crest_factor": crest
        })

    except Exception as e:
        print(f"[ERROR] Could not process {wav_file}: {e}")

# =======================
# SAVE RESULTS
# =======================
df = pd.DataFrame(file_stats)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved detailed quality stats to {OUTPUT_CSV}")

# =======================
# SUMMARY
# =======================
print(f"\n🌬 WIND DATASET QUALITY SUMMARY")
print(f"Total files: {len(df)}")
print(f"Total duration: {df['duration_s'].sum() / 3600:.2f} hours")
print(f"RMS: min={df['rms'].min():.4f}, max={df['rms'].max():.4f}, mean={df['rms'].mean():.4f}")
print(f"Silence ratio: min={df['silence_ratio'].min():.2f}, max={df['silence_ratio'].max():.2f}, mean={df['silence_ratio'].mean():.2f}")
print(f"Spectral centroid (Hz): min={df['spectral_centroid'].min():.2f}, max={df['spectral_centroid'].max():.2f}, mean={df['spectral_centroid'].mean():.2f}")
print(f"Spectral flatness: min={df['spectral_flatness'].min():.4f}, max={df['spectral_flatness'].max():.4f}, mean={df['spectral_flatness'].mean():.4f}")
print(f"Zero crossing rate: min={df['zero_crossing_rate'].min():.4f}, max={df['zero_crossing_rate'].max():.4f}, mean={df['zero_crossing_rate'].mean():.4f}")
print(f"Dynamic range: min={df['dynamic_range'].min():.4f}, max={df['dynamic_range'].max():.4f}, mean={df['dynamic_range'].mean():.4f}")
print(f"Crest factor: min={df['crest_factor'].min():.4f}, max={df['crest_factor'].max():.4f}, mean={df['crest_factor'].mean():.4f}")

