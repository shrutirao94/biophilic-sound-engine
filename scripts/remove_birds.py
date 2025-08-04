import os
import shutil

# === CONFIG ===
base_path = "/Users/shrutirao/Documents/projects/study-3/biophilic-sound-engine/data/processed_classes_filtered"
weak_clips_file = "/Users/shrutirao/Documents/projects/study-3/biophilic-sound-engine/weak_clips.txt"  # path to weak_clips.txt
source_folder = os.path.join(base_path, "birds")
destination_folder = os.path.join(base_path, "check_files")

# Read weak clips list
with open(weak_clips_file, "r") as f:
    weak_files = [line.strip() for line in f if line.strip()]

copied_files = []
missing_files = []

for file_path in weak_files:
    # Remove "data/raw/processed_classes_filtered" from the path in weak_clips.txt
    relative_path = os.path.relpath(file_path, "data/raw/processed_classes_filtered")

    source_file = os.path.join(base_path, relative_path)
    destination_file = os.path.join(destination_folder, relative_path)

    if os.path.exists(source_file):
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied_files.append(relative_path)
    else:
        missing_files.append(relative_path)

print(f"Copied {len(copied_files)} files to {destination_folder}")
if missing_files:
    print(f"Missing files ({len(missing_files)}):")
    for mf in missing_files:
        print(mf)

