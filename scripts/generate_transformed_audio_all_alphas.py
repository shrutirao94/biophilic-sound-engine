# import os
# import pandas as pd
# from pydub import AudioSegment
# from apply_transformation_effects import apply_transformation_effects

# # === SETTINGS ===
# SEGMENT_DIR = "data/segments/office/"
# ORIGINAL_FEATURES_PATH = "features/office/extracted-features.csv"
# TRANSFORMED_DIR = "data/transformed/"
# OUTPUT_DIR = "data/processed_transformed/"
# ALPHAS = [0.8, 1.0]

# # Ensure output folder exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load original feature data
# original_df = pd.read_csv(ORIGINAL_FEATURES_PATH)
# use_filenames = 'filename' in original_df.columns

# # === Loop through alpha versions ===
# for alpha in ALPHAS:
    # alpha_str = f"{alpha:.1f}"
    # transformed_path = os.path.join(TRANSFORMED_DIR, f"office_alpha_{alpha_str}.csv")
    
    # if not os.path.exists(transformed_path):
        # print(f"❌ ransformed CSV missing for alpha={alpha_str}")
        # continue
    
    # transformed_df = pd.read_csv(transformed_path)

    # print(f"\n🎧 Processing alpha={alpha_str}...")
    
    # for idx in range(len(original_df)):
        # if use_filenames:
            # fname = original_df.iloc[idx]['filename']
        # else:
            # fname = f"segment_{idx:04d}.wav"

        # segment_path = os.path.join(SEGMENT_DIR, fname)
        # if not os.path.exists(segment_path):
            # print(f"⚠️ Skipping missing file: {fname}")
            # continue

        # audio = AudioSegment.from_wav(segment_path)
        # orig_feats = original_df.iloc[idx].to_dict()
        # trans_feats = transformed_df.iloc[idx].to_dict()

        # # Apply transformation effects
        # modified_audio = apply_transformation_effects(audio, orig_feats, trans_feats)

        # # Save output with alpha in filename
        # outname = fname.replace(".wav", f"_alpha_{alpha_str}_transformed.wav")
        # outpath = os.path.join(OUTPUT_DIR, outname)
        # modified_audio.export(outpath, format="wav")

    # print(f"✅ Finished alpha={alpha_str}")


import os
import pandas as pd
from pydub import AudioSegment
from apply_transformation_effects import apply_transformation_effects

# === SETTINGS ===
SEGMENT_DIR = "data/segments/office/"
ORIGINAL_FEATURES_PATH = "features/office/extracted-features.csv"
TRANSFORMED_DIR = "data/transformed/"
OUTPUT_DIR = "data/processed_transformed/"
ALPHAS = [0.8, 1.0]

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original feature data
original_df = pd.read_csv(ORIGINAL_FEATURES_PATH)
use_filenames = 'filename' in original_df.columns

# === Loop through alpha versions ===
for alpha in ALPHAS:
    alpha_str = f"{alpha:.1f}"
    transformed_path = os.path.join(TRANSFORMED_DIR, f"office_alpha_{alpha_str}.csv")

    if not os.path.exists(transformed_path):
        print(f"❌ Transformed CSV missing for alpha={alpha_str}")
        continue

    transformed_df = pd.read_csv(transformed_path)
    print(f"\n🎧 Processing alpha={alpha_str}...")

    for idx in range(len(original_df)):
        # Determine filename
        if use_filenames:
            fname = original_df.iloc[idx]['filename']
        else:
            fname = f"segment_{idx:04d}.wav"

        segment_path = os.path.join(SEGMENT_DIR, fname)
        if not os.path.exists(segment_path):
            print(f"⚠️ Skipping missing file: {fname}")
            continue

        # Load and transform
        audio = AudioSegment.from_wav(segment_path)
        orig_feats = original_df.iloc[idx].to_dict()
        trans_feats = transformed_df.iloc[idx].to_dict()
        modified_audio = apply_transformation_effects(audio, orig_feats, trans_feats)

        # Save output
        outname = fname.replace(".wav", f"_alpha_{alpha_str}_transformed.wav")
        outpath = os.path.join(OUTPUT_DIR, outname)
        modified_audio.export(outpath, format="wav")

    print(f"✅ Finished alpha={alpha_str}")

