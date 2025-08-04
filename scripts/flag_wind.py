import os
import shutil
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Input folder
WIND_DIR = Path("data/processed_classes_filtered/wind")
OUTPUT_CSV = WIND_DIR / "wind_filtered_stats.csv"

# Thresholds for filtering
THRESHOLDS = {
    "silence_ratio": 0.5,
    "rms_low": 0.02,
    "rms_high": 0.25,
    "spectral_centroid_high": 4000,
    "crest_factor_high": 20,
    "dynamic_range_low": 0.05,
}

# Helper: calculate silence ratio
def calculate_silence_ratio(y, threshold=0.01):
    return np.mean(np.abs(y) < threshold)

# Helper: calculate dynamic range
def calculate_dynamic_range(y):
    return np.percentile(np.abs(y), 95) - np.percentile(np.abs(y), 5)

# Helper: calculate crest factor
def calculate_crest_factor(y):
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(y**2))
    return peak / rms if rms > 0 else 0

# Create subfolder if not exists
def move_file(file_path, subfolder):
    target_dir = WIND_DIR / subfolder
    target_dir.mkdir(exist_ok=True)
    shutil.move(str(file_path), target_dir / file_path.name)

file_stats = []

print("🔍 Analyzing wind dataset quality and moving flagged files...")

# Analyze all wind files
for wav_file in WIND_DIR.glob("*"):
    if wav_file.suffix.lower() not in [".wav", ".mp3", ".flac"]:
        continue

    try:
        y, sr = librosa.load(wav_file, sr=48000, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        silence_ratio = calculate_silence_ratio(y)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        dynamic_range = calculate_dynamic_range(y)
        crest_factor = calculate_crest_factor(y)

        # Flagging
        flags = []
        if silence_ratio > THRESHOLDS["silence_ratio"]:
            flags.append("silent")
        if rms < THRESHOLDS["rms_low"]:
            flags.append("quiet")
        elif rms > THRESHOLDS["rms_high"]:
            flags.append("loud")
        if spectral_centroid > THRESHOLDS["spectral_centroid_high"]:
            flags.append("high_centroid")
        if crest_factor > THRESHOLDS["crest_factor_high"]:
            flags.append("crest_spikes")
        if dynamic_range < THRESHOLDS["dynamic_range_low"]:
            flags.append("low_dynamic_range")

        # Move files if flagged
        for flag in flags:
            move_file(wav_file, flag)

        file_stats.append({
            "filename": wav_file.name,
            "duration_s": duration,
            "rms": rms,
            "silence_ratio": silence_ratio,
            "spectral_centroid": spectral_centroid,
            "spectral_flatness": spectral_flatness,
            "zero_crossing_rate": zcr,
            "dynamic_range": dynamic_range,
            "crest_factor": crest_factor,
            "flags": ", ".join(flags) if flags else "ok"
        })

    except Exception as e:
        print(f"[ERROR] Could not process {wav_file}: {e}")

# Save results to CSV
df = pd.DataFrame(file_stats)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved detailed quality stats and flags to {OUTPUT_CSV}")
print(f"🚚 Flagged files have been moved into subfolders under {WIND_DIR}")
print("\n📊 Summary of flagged files:")
print(df["flags"].value_counts())

