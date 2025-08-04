from pathlib import Path
import librosa
import numpy as np

# Path to your wind dataset
WIND_DIR = Path("data/processed_classes_filtered/wind/")

file_durations = []
errors = 0

for wav_file in WIND_DIR.glob("*.wav"):
    try:
        y, sr = librosa.load(wav_file, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        file_durations.append(duration)
    except Exception as e:
        print(f"[ERROR] Could not load {wav_file}: {e}")
        errors += 1

# Dataset stats
total_files = len(file_durations)
total_hours = np.sum(file_durations) / 3600 if file_durations else 0
avg_duration = np.mean(file_durations) if file_durations else 0
min_duration = np.min(file_durations) if file_durations else 0
max_duration = np.max(file_durations) if file_durations else 0

print("\n🌬  DATASET SUMMARY")
print(f"Total files: {total_files}")
print(f"Total duration: {total_hours:.2f} hours")
print(f"Average file length: {avg_duration:.2f} s")
print(f"Shortest file: {min_duration:.2f} s")
print(f"Longest file: {max_duration:.2f} s")
print(f"Failed files: {errors}")

