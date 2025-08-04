import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# --- CONFIG ---
AUDIO_DIR = Path("data/processed_classes_filtered/wind")  # source directory
OUTLIER_DIR = Path("data/outliers/wind")       # where outliers will be moved
OUTLIER_DIR.mkdir(parents=True, exist_ok=True)

# --- FEATURE EXTRACTION ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    return rms, centroid, flatness, zcr

# --- SCAN FILES ---
files = list(AUDIO_DIR.rglob("*.wav"))
print(f"Found {len(files)} wav files in {AUDIO_DIR}")

if not files:
    exit("⚠ No files found. Check folder path!")

data = []
for f in files:
    try:
        rms, centroid, flatness, zcr = extract_features(f)
        data.append({"file": f, "rms": rms, "centroid": centroid, "flatness": flatness, "zcr": zcr})
    except Exception as e:
        print(f"Error processing {f}: {e}")

df = pd.DataFrame(data)

# --- OUTLIER DETECTION (Z-SCORE) ---
outliers = set()
for feature in ["rms", "centroid", "flatness", "zcr"]:
    mean = df[feature].mean()
    std = df[feature].std()
    if std == 0:
        continue
    feature_outliers = df[(df[feature] < mean - 3 * std) | (df[feature] > mean + 3 * std)]
    outliers.update(feature_outliers.index)

outlier_files = df.loc[list(outliers), "file"]

print(f"🚨 Found {len(outlier_files)} outliers")

# --- MOVE OUTLIERS ---
for file_path in outlier_files:
    dest = OUTLIER_DIR / file_path.name
    shutil.move(file_path, dest)
    print(f"Moved: {file_path.name} → {dest}")

print(f"✅ Done. Outliers moved to {OUTLIER_DIR}")

