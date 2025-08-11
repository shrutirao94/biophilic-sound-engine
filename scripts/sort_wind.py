import librosa
import numpy as np
from pathlib import Path
import shutil

# === PATHS ===
base_dir = Path("/Users/shrutirao/Documents/projects/study-3/biophilic-sound-engine/data/processed_classes_filtered/wind")
real_dir = base_dir / "real"
synth_dir = base_dir / "synthetic"

# Create folders if they don't exist
real_dir.mkdir(exist_ok=True)
synth_dir.mkdir(exist_ok=True)

# === PARAMETERS ===
HNR_THRESHOLD = 10.0  # dB; tweak this if needed

def compute_hnr(file_path):
    """Compute Harmonic-to-Noise Ratio for an audio file."""
    y, sr = librosa.load(file_path, sr=48000, mono=True)
    harmonic, _ = librosa.effects.hpss(y)
    noise = y - harmonic
    hnr = 10 * np.log10((np.sum(harmonic**2) + 1e-8) / (np.sum(noise**2) + 1e-8))
    return hnr

# === PROCESS FILES ===
for wav_file in base_dir.glob("*.wav"):
    hnr = compute_hnr(wav_file)

    if hnr <= HNR_THRESHOLD:
        target = real_dir / wav_file.name
    else:
        target = synth_dir / wav_file.name

    print(f"{wav_file.name}: HNR={hnr:.2f} dB → {'real' if hnr <= HNR_THRESHOLD else 'synthetic'}")
    shutil.move(str(wav_file), str(target))

print("\n✅ Sorting complete! Files moved into 'real' and 'synthetic' folders.")

