# scripts/segment_curated_nature.py

import os
from pathlib import Path
import librosa
import soundfile as sf

SAMPLE_RATE = 48000
SEGMENT_DURATION = 60  # in seconds
MIN_SEGMENT_LENGTH = 1.0  # in seconds

RAW_DIR = Path("data/raw/curated_nature")
OUT_DIR = Path("data/processed_curated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

for category_dir in RAW_DIR.iterdir():
    if not category_dir.is_dir():
        continue

    category = category_dir.name
    out_category_dir = OUT_DIR / category
    out_category_dir.mkdir(parents=True, exist_ok=True)

    for audio_file in category_dir.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        try:
            y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
            y = y / max(abs(y)) if max(abs(y)) > 0 else y

            total_samples = len(y)
            segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)

            for i in range(0, total_samples, segment_samples):
                segment = y[i:i+segment_samples]
                duration = len(segment) / SAMPLE_RATE

                if duration < MIN_SEGMENT_LENGTH:
                    continue

                seg_name = f"{audio_file.stem}_seg{i // segment_samples:04d}.wav"
                out_path = out_category_dir / seg_name
                sf.write(out_path, segment, SAMPLE_RATE)

            print(f"✅ Processed: {audio_file.name} → {category}/{seg_name}")

        except Exception as e:
            print(f"[ERROR] {audio_file.name}: {e}")

