import os
import pandas as pd
from pydub import AudioSegment
from apply_transformation_effects import apply_transformation_effects

# Settings
SEGMENT_DIR = "data/segments/office/"
OUTPUT_DIR = "data/processed_transformed/"
ORIGINAL_FEATURES_PATH = "features/office/extracted-features.csv"
TRANSFORMED_FEATURES_PATH = "data/transformed/office_alpha_0.3.csv"  # Adjust alpha version

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load feature data
original_df = pd.read_csv(ORIGINAL_FEATURES_PATH)
transformed_df = pd.read_csv(TRANSFORMED_FEATURES_PATH)

# Optionally identify by filename if available
use_filenames = 'filename' in original_df.columns

# Process each segment
for idx in range(len(original_df)):
    if use_filenames:
        fname = original_df.iloc[idx]['filename']
    else:
        fname = f"segment_{idx:04d}.wav"

    segment_path = os.path.join(SEGMENT_DIR, fname)
    if not os.path.exists(segment_path):
        print(f"Skipping missing file: {fname}")
        continue

    audio = AudioSegment.from_wav(segment_path)
    
    # Get original and transformed feature rows as dicts
    orig_feats = original_df.iloc[idx].to_dict()
    trans_feats = transformed_df.iloc[idx].to_dict()

    # Apply transformation effects
    modified_audio = apply_transformation_effects(audio, orig_feats, trans_feats)

    # Save output
    outname = fname.replace(".wav", f"_alpha_0.3_transformed.wav")
    outpath = os.path.join(OUTPUT_DIR, outname)
    modified_audio.export(outpath, format="wav")
    print(f" Transformed and saved: {outname}")

