import pandas as pd
import numpy as np
import os

# Paths (adjust as needed)
OFFICE_FEATURE_PATH = "features/office/extracted-features.csv"
NATURE_FEATURE_PATH = "features/nature/extracted-features.csv"
OUTPUT_DIR = "data/transformed/"

# Load data
def load_features(path):
    return pd.read_csv(path)

# # Compute statistical means of nature dataset
# def compute_nature_means(nature_df):
    # return nature_df.mean(numeric_only=True)

 # Compute per-feature statistical targets for transformation
def compute_nature_targets(nature_df):
    return {
        'rms': nature_df['rms'].median(),
        'centroid': nature_df['centroid'].mean(),
        'flatness': nature_df['flatness'].median(),
        'bandwidth': nature_df['bandwidth'].mean(),
        'zcr': nature_df['zcr'].median(),
        'onset_density': nature_df['onset_density'].mean()
    }

# Perform linear transformation towards target values
def linear_transform(office_feature, nature_target, alpha=0.2):
    return office_feature + alpha * (nature_target - office_feature)

# Transform office features
def transform_features(office_df, nature_targets, alpha=0.2):
    transformed_df = office_df.copy()
    for feature, target_value in nature_targets.items():
        if feature in office_df.columns:
            transformed_df[feature] = linear_transform(office_df[feature], target_value, alpha)
    return transformed_df

# Main function
def main():
    office_df = load_features(OFFICE_FEATURE_PATH)
    nature_df = load_features(NATURE_FEATURE_PATH)
    nature_targets = compute_nature_targets(nature_df)

    alphas = [0.1, 0.3, 0.5, 0.8, 1.0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for alpha in alphas:
        transformed_df = transform_features(office_df, nature_targets, alpha)
        output_path = os.path.join(OUTPUT_DIR, f"office_alpha_{alpha:.1f}.csv")
        transformed_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

