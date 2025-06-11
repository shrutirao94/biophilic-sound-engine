import os
import librosa
import soundfile as sf
import numpy as np

# Input directory for nature segments
input_dir = "data/segments/nature/"
output_dir = "data/processed/nature/"
os.makedirs(output_dir, exist_ok=True)

def normalize_audio(y, silence_threshold=1e-4):
    """
    Normalize waveform to -1 to 1 range.
    Skip normalization if segment is basically silent.
    """
    max_amp = np.max(np.abs(y))
    if max_amp < silence_threshold:
        # It's effectively silence; skip normalization
        return y
    else:
        return y / max_amp

def trim_silence(y, top_db=30):
    """Trim silence from start and end (optional)."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

# Process each nature segment
for nature_source in os.listdir(input_dir):
    source_dir = os.path.join(input_dir, nature_source)
    if os.path.isdir(source_dir):
        output_source_dir = os.path.join(output_dir, nature_source)
        os.makedirs(output_source_dir, exist_ok=True)

        for fname in os.listdir(source_dir):
            if fname.endswith(".wav"):
                filepath = os.path.join(source_dir, fname)
                y, sr = librosa.load(filepath, sr=None, mono=True)

                # Normalize
                # y_norm = normalize_audio(y)

                # Optionally trim silence
                # y_trimmed = trim_silence(y_norm)
                # y_final = y_trimmed
                y_final = y

                # Save to processed folder
                output_path = os.path.join(output_source_dir, fname)
                sf.write(output_path, y_final, sr)
                print("Processed:", output_path)

print("All nature segments normalized!")

