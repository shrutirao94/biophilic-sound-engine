import os
import librosa
import soundfile as sf

# Parameters
input_file = "data/raw/office.mp3"  # Path to your WAV file
output_dir = "data/segments"
segment_length = 30  # seconds

# Load audio file
y, sr = librosa.load(input_file, sr=None, mono=True)
duration = librosa.get_duration(y=y, sr=sr)

print("Sample Rate:", sr)
print("Total Duration (seconds):", duration)

# Calculate number of segments
num_segments = int(duration // segment_length)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Segment and save
for i in range(num_segments):
    start_sample = int(i * segment_length * sr)
    end_sample = int((i + 1) * segment_length * sr)
    segment = y[start_sample:end_sample]
    output_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
    sf.write(output_path, segment, sr)
    print(f"Saved: {output_path}")

print("Segmentation complete!")

