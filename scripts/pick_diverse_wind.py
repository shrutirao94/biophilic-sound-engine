import librosa
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shutil
import os

# === CONFIG ===
SOURCE_DIR = Path("data/raw/curated_nature/wind/new/")
OUTPUT_DIR = Path("data/raw/diverse_wind")
N_CLUSTERS = 5         # Number of clusters to group wind files
FILES_PER_CLUSTER = 3  # How many to copy per cluster (adjust if needed)

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

features = []
files = []

print("🔍 Extracting audio features...")
for file in SOURCE_DIR.glob("*.wav"):
    y, sr = librosa.load(file, sr=None, mono=True)
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append([rms, centroid, bandwidth, flatness, zcr])
    files.append(file)

features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("📊 Running KMeans clustering...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
labels = kmeans.fit_predict(features_scaled)

print(f"✅ Clustering complete. {N_CLUSTERS} clusters identified.")

selected_files = []
for cluster in range(N_CLUSTERS):
    cluster_indices = np.where(labels == cluster)[0]
    center = kmeans.cluster_centers_[cluster]

    # Sort files by distance to cluster center (closest = most representative)
    distances = np.linalg.norm(features_scaled[cluster_indices] - center, axis=1)
    sorted_indices = cluster_indices[np.argsort(distances)]

    # Pick top N files per cluster
    for idx in sorted_indices[:FILES_PER_CLUSTER]:
        selected_files.append(files[idx])

# Copy selected files to new folder
print(f"📂 Copying {len(selected_files)} selected diverse files to {OUTPUT_DIR}...")
for f in selected_files:
    shutil.copy(f, OUTPUT_DIR / f.name)

print("✅ Done! Diverse wind files copied.")
print(f"Total selected: {len(selected_files)}")
print(f"Saved in: {OUTPUT_DIR}")

