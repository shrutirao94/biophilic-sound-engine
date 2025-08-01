import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import shutil

# === CONFIG ===
main_dir = Path("data/processed_classes_filtered/wind")
short_dir = main_dir / "short"
target_duration = 60  # seconds

# Create a list of short files
short_files = [f for f in os.listdir(short_dir) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"))]

for file in short_files:
    file_path = short_dir / file
    y, sr = librosa.load(file_path, sr=None, mono=True)

    current_duration = librosa.get_duration(y=y, sr=sr)

    if current_duration < target_duration:
        # Calculate how many times to repeat the clip
        repeat_count = int(np.ceil(target_duration / current_duration))
        y_looped = np.tile(y, repeat_count)  # Repeat waveform
        y_looped = y_looped[: int(target_duration * sr)]  # Trim to exactly 60s

        # Save to main wind folder
        output_path = main_dir / file.replace(".mp3", ".wav").replace(".flac", ".wav")
        sf.write(output_path, y_looped, sr)
        print(f"🔁 Looped: {file} ({current_duration:.2f}s → 60s)")

    else:
        # If it's already >= 60s (edge case), just move it
        output_path = main_dir / file
        shutil.move(file_path, output_path)
        print(f"➡️ Moved without looping: {file}")

print(f"\n✅ Completed looping and moving all short files into {main_dir}")

