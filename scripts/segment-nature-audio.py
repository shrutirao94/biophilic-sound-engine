import os
import librosa
import soundfile as sf

# Input and output directories
nature_dir = "data/raw/nature/"
output_dir = "data/segments/nature/"
segment_duration = 30  # seconds

# Get list of MP3 files
nature_files = [f for f in os.listdir(nature_dir) if f.endswith(".mp3")]

for nature_file in nature_files:
    input_path = os.path.join(nature_dir, nature_file)
    y, sr = librosa.load(input_path, sr=None)
    
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(total_duration // segment_duration)
    
    # Create output directory for this file's segments
    file_base = os.path.splitext(nature_file)[0]
    file_output_dir = os.path.join(output_dir, file_base)
    os.makedirs(file_output_dir, exist_ok=True)
    
    for i in range(num_segments):
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr)
        segment_audio = y[start_sample:end_sample]
        
        segment_filename = f"segment-{i+1:03d}.wav"
        segment_path = os.path.join(file_output_dir, segment_filename)
        sf.write(segment_path, segment_audio, sr)
    
    print(f"Segmented {nature_file} into {num_segments} files.")

