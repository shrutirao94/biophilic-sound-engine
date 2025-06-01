import os
import librosa
import soundfile as sf
import numpy as np

# Parameters
input_dir = "data/segments"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

def normalize_audio(y):
    """Normalize to -1.0 to 1.0."""
    return y / np.max(np.abs(y))

def trim_silence(y, top_db=30):
    """Trim silence from start and end (optional)."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

# Process each segment
for fname in os.listdir(input_dir):
    if fname.endswith(".wav"):
        filepath = os.path.join(input_dir, fname)
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Normalize
        y_norm = normalize_audio(y)
        
        # Optionally trim silence
        # y_trimmed = trim_silence(y_norm)
        # y_final = y_trimmed
        y_final = y_norm
        
        # Save to processed folder
        output_path = os.path.join(output_dir, fname)
        sf.write(output_path, y_final, sr)
        print("Processed:", output_path)

print("All segments normalized!")

