import os
import librosa
import soundfile as sf
import numpy as np

# Parameters
input_dir = "data/segments/office/"
output_dir = "data/processed/office/"
os.makedirs(output_dir, exist_ok=True)


def rms(y):
    """Compute the RMS value of an audio signal."""
    return np.sqrt(np.mean(y ** 2))


def normalize_audio_rms(y, silence_threshold=1e-4):
    """
    Normalize waveform using RMS-based approach.
    Skip normalization if RMS is below threshold.
    """
    current_rms = rms(y)
    if current_rms < silence_threshold:
        return y
    else:
        target_rms = 0.1
        gain = target_rms / current_rms
        return y * gain


def trim_silence(y, top_db=30):
    """Trim silence from start and end (optional)."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

# Process each segment
for fname in os.listdir(input_dir):
    if fname.endswith(".wav"):
        filepath = os.path.join(input_dir, fname)
        y, sr = librosa.load(filepath, sr=None, mono=True)

        # Normalize using RMS-based method
        # y_norm = normalize_audio_rms(y)

        # Optionally trim silence
        # y_trimmed = trim_silence(y_norm)
        # y_final = y_trimmed
        y_final = y

        # Save to processed folder
        output_path = os.path.join(output_dir, fname)
        sf.write(output_path, y_final, sr)
        print("Processed:", output_path)

print("All segments normalized!")

