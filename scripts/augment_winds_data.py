from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = Path("data/raw/curated_nature/wind")
OUTPUT_DIR = Path("data/raw/augmented_wind")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_LENGTH = 60  # seconds
AUGMENTATIONS = [
    {"time_stretch": 1.0, "pitch": 0},     # original
    {"time_stretch": 1.05, "pitch": 0},    # slight speed up
    {"time_stretch": 0.95, "pitch": 0},    # slight slow down
    {"time_stretch": 1.0, "pitch": 0.5},   # slight pitch up
    {"time_stretch": 1.0, "pitch": -0.5},  # slight pitch down
]

for file in INPUT_DIR.glob("*.wav"):
    y, sr = librosa.load(file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Split long files into 60s chunks
    num_segments = max(1, int(np.ceil(duration / SEGMENT_LENGTH)))
    for s in range(num_segments):
        start = int(s * SEGMENT_LENGTH * sr)
        end = int(min((s + 1) * SEGMENT_LENGTH * sr, len(y)))
        segment = y[start:end]

        for aug in AUGMENTATIONS:
            aug_y = librosa.effects.time_stretch(segment, rate=aug["time_stretch"])
            aug_y = librosa.effects.pitch_shift(aug_y, sr=sr, n_steps=aug["pitch"])

            # Random small gain adjustment
            gain = np.random.uniform(0.95, 1.05)
            aug_y *= gain

            out_path = OUTPUT_DIR / f"{file.stem}_seg{s+1}_ts{aug['time_stretch']}_p{aug['pitch']}.wav"
            sf.write(out_path, aug_y, sr)

print("✅ Augmentation complete! Files saved in:", OUTPUT_DIR)

