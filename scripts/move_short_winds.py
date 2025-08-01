import os
import shutil
import librosa
from pathlib import Path

# === CONFIG ===
input_dir = Path("data/processed_classes_filtered/wind")  # Your segmented wind dataset
short_dir = input_dir / "short"  # Folder for short clips
threshold = 60  # seconds

short_dir.mkdir(parents=True, exist_ok=True)

# Supported audio formats
SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

moved_count = 0

for file in os.listdir(input_dir):
    file_path = input_dir / file

    # Skip directories and non-audio files
    if file_path.is_dir() or not file.lower().endswith(tuple(ext.lower() for ext in SUPPORTED_FORMATS)):
        continue

    # Load audio file to check duration
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < threshold:
        dest_path = short_dir / file
        shutil.move(file_path, dest_path)
        moved_count += 1
        print(f"📂 Moved: {file} ({duration:.2f}s) → short/")

print(f"\n✅ Moved {moved_count} files shorter than {threshold}s to '{short_dir}'")

