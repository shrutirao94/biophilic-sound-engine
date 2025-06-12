import pandas as pd
import numpy as np
import os

# Paths (adjust as needed)
OFFICE_FEATURE_PATH = "features/office/extracted-features.csv"
NATURE_FEATURE_PATH = "features/nature/extracted-features.csv"
OUTPUT_PATH = "data/transformed/office-features-transformed.csv"

# Load data
def load_features(path):
    return pd.read_csv(path)

# Compute statistical means of nature dataset
def compute_nature_means(nature_df):
    return nature_df.mean(numeric_only=True)

# Perform linear transformation towards nature means
def linear_transform(office_feature, nature_mean, alpha=0.2):
    return office_feature + alpha * (nature_mean - office_feature)

# Transform office features
def transform_features(office_df, nature_means, alpha=0.2):
    transformed_df = office_df.copy()
    feature_columns = nature_means.index
    for feature in feature_columns:
        if feature in office_df.columns:
            transformed_df[feature] = linear_transform(office_df[feature], nature_means[feature], alpha)
    return transformed_df

# Main function
def main(alpha=0.2):
    office_df = load_features(OFFICE_FEATURE_PATH)
    nature_df = load_features(NATURE_FEATURE_PATH)

    nature_means = compute_nature_means(nature_df)

    transformed_df = transform_features(office_df, nature_means, alpha)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save transformed features
    transformed_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Transformation complete. Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main(alpha=0.2)  # Adjust alpha as needed
