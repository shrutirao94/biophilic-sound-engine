# scripts/check_ready_for_rave.py
from pathlib import Path
import soundfile as sf
import numpy as np
from collections import defaultdict

DATA_DIR = Path("data/processed_classes_filtered/water_ready")

# Tolerances / limits
TARGET_SR = 48000
TARGET_CHANNELS = 1
TARGET_DUR = 60.0
DUR_TOL = 0.02          # +/- 20 ms
CLIP_PEAK_MAX = 0.9995  # consider anything above this as clipped
DC_OFFSET_MAX = 0.01    # absolute mean

SKIP_DIRS = {"review_flags", "augmented_60s", "backups_short60"}

def iter_audio_files(root: Path):
    for p in sorted(root.glob("*.wav")):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        yield p

def analyze_file(path: Path):
    """Fast header checks via sf.info + streaming peak & dc offset."""
    info = sf.info(path)
    sr = info.samplerate
    ch = info.channels
    frames = info.frames
    dur = frames / sr if sr else 0.0

    # Stream to get peak/DC without loading all into memory
    peak = 0.0
    dc_sum = 0.0
    n_samples = 0
    with sf.SoundFile(path, "r") as f:
        for block in f.blocks(blocksize=65536, dtype="float32", always_2d=True):
            # Mix to mono for stats (even if file is mono already)
            mono = block.mean(axis=1)
            peak = max(peak, float(np.max(np.abs(mono)))) if mono.size else peak
            dc_sum += float(mono.sum())
            n_samples += mono.size
    dc_offset = (dc_sum / n_samples) if n_samples else 0.0

    return {
        "file": path,
        "sr": sr,
        "channels": ch,
        "duration": dur,
        "peak": peak,
        "dc_offset": dc_offset,
    }

def main():
    files = list(iter_audio_files(DATA_DIR))
    issues = defaultdict(list)
    ok_count = 0

    for p in files:
        m = analyze_file(p)

        bad = False
        if m["sr"] != TARGET_SR:
            issues["wrong_sr"].append((p, m["sr"]))
            bad = True
        if m["channels"] != TARGET_CHANNELS:
            issues["not_mono"].append((p, m["channels"]))
            bad = True
        if not (TARGET_DUR - DUR_TOL <= m["duration"] <= TARGET_DUR + DUR_TOL):
            issues["wrong_duration"].append((p, m["duration"]))
            bad = True
        if m["peak"] > CLIP_PEAK_MAX:
            issues["clipping_peak"].append((p, m["peak"]))
            bad = True
        if abs(m["dc_offset"]) > DC_OFFSET_MAX:
            issues["dc_offset"].append((p, m["dc_offset"]))
            bad = True

        if not bad:
            ok_count += 1

    total = len(files)
    print("\n===== RAVE READINESS CHECK (water) =====")
    print(f"Checked files: {total}")
    print(f"Pass (ready):  {ok_count}")
    fail = total - ok_count
    print(f"Fail (needs fix): {fail}")

    if issues:
        print("\n--- Issues by type ---")
        for k, lst in issues.items():
            print(f"{k}: {len(lst)}")
        # Print a few examples per issue
        print("\n--- Examples (up to 5 each) ---")
        for k, lst in issues.items():
            print(f"\n{k}:")
            for i, (p, v) in enumerate(lst[:5]):
                print(f"  - {p.name}  →  {v}")

    # Optional: write a simple manifest of non-compliant files
    if issues:
        manifest = DATA_DIR / "not_ready_for_rave.txt"
        with open(manifest, "w") as f:
            for k, lst in issues.items():
                f.write(f"[{k}]\n")
                for p, v in lst:
                    f.write(f"{p}\t{v}\n")
        print(f"\nWrote list of non-compliant files → {manifest}")

if __name__ == "__main__":
    main()

