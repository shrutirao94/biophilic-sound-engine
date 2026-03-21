import os
import librosa
import soundfile as sf
from pathlib import Path

# === CONFIG ===
input_dir = Path("data/raw/curated_nature/water")               # Input folder
output_dir = Path("data/processed_classes_filtered/water") # Output folder
segment_duration = 60  # seconds

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")

output_dir.mkdir(parents=True, exist_ok=True)

# Process all audio files
audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(tuple(ext.lower() for ext in SUPPORTED_FORMATS))]

for audio_file in audio_files:
    input_path = input_dir / audio_file
    y, sr = librosa.load(input_path, sr=None, mono=True)

    total_duration = librosa.get_duration(y=y, sr=sr)

    # If file is <= 60s, keep as is (copy to output folder)
    if total_duration <= segment_duration:
        output_path = output_dir / f"{Path(audio_file).stem}.wav"
        sf.write(output_path, y, sr)
        print(f"✅ Kept {audio_file} (no split, {total_duration:.2f}s)")
        continue

    # Split into 60s chunks if > 60s
    num_segments = int(total_duration // segment_duration)

    for i in range(num_segments):
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr)
        segment_audio = y[start_sample:end_sample]

        segment_filename = f"{Path(audio_file).stem}-part{i+1:03d}.wav"
        segment_path = output_dir / segment_filename
        sf.write(segment_path, segment_audio, sr)

    # Add remainder if > 0s
    remainder = total_duration % segment_duration
    if remainder > 0:
        start_sample = int(num_segments * segment_duration * sr)
        remainder_audio = y[start_sample:]
        remainder_filename = f"{Path(audio_file).stem}-part{num_segments+1:03d}.wav"
        remainder_path = output_dir / remainder_filename
        sf.write(remainder_path, remainder_audio, sr)

    print(f"✅ Split {audio_file} into {num_segments}x60s + remainder ({remainder:.2f}s)")

