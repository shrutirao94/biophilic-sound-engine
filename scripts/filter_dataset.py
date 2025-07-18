# scripts/filter_by_category.py
import os
import librosa
import shutil
import soundfile as sf
import numpy as np
from tqdm import tqdm

INPUT_ROOT = "data/processed_curated/"
OUTPUT_ROOT = "data/processed_curated_filtered/"
CHECK_ROOT = "data/processed_curated_check/"

# Thresholds
RMS_THRESHOLD = 0.0015  # relaxed threshold
STD_THRESHOLD = 0.003
SPECTRAL_FLATNESS_THRESHOLD = 0.3

def is_silent(y):
    rms = np.mean(librosa.feature.rms(y=y))
    std = np.std(y)
    return rms < RMS_THRESHOLD or std < STD_THRESHOLD

def is_static_like(y):
    flatness = librosa.feature.spectral_flatness(y=y)
    return np.mean(flatness) > SPECTRAL_FLATNESS_THRESHOLD

# Create output/check folders
for root in [OUTPUT_ROOT, CHECK_ROOT]:
    for cat in os.listdir(INPUT_ROOT):
        os.makedirs(os.path.join(root, cat, "silent"), exist_ok=True)
        os.makedirs(os.path.join(root, cat, "static"), exist_ok=True)

# Walk through categories
for category in sorted(os.listdir(INPUT_ROOT)):
    category_path = os.path.join(INPUT_ROOT, category)
    if not os.path.isdir(category_path):
        continue

    print(f"\n🔍 Processing category: {category}")
    files = [f for f in os.listdir(category_path) if f.endswith(".wav")]

    for fname in tqdm(files, desc=f"{category:>10}"):
        input_path = os.path.join(category_path, fname)

        try:
            y, sr = librosa.load(input_path, sr=None, mono=True)

            if is_silent(y):
                shutil.move(input_path, os.path.join(CHECK_ROOT, category, "silent", fname))
            elif is_static_like(y):
                shutil.move(input_path, os.path.join(CHECK_ROOT, category, "static", fname))
            else:
                out_path = os.path.join(OUTPUT_ROOT, category, fname)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.copy(input_path, out_path)

        except Exception as e:
            print(f"❌ [ERROR] {fname}: {e}")

