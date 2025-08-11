# scripts/make_short_segments_60s_review.py
import math
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

DATA_DIR = Path("data/processed_classes_filtered/water")
AUG_DIR  = DATA_DIR / "augmented_60s"   # <- new 60s versions go here
TARGET_SEC = 60.0
TARGET_SR  = 48000
MODE = "loop_xfade"                     # "loop_xfade" (recommended) or "pad_silence"
XFADE_MS = 150                          # crossfade for loop mode
FADE_MS  = 10                           # tiny fade in/out to avoid clicks
WRITE_SUBTYPE = "PCM_16"                # or "PCM_24" / "FLOAT"

AUG_DIR.mkdir(parents=True, exist_ok=True)

def tiny_fade(y, sr, fade_ms=10):
    n = len(y)
    f = int(sr * fade_ms / 1000.0)
    f = max(1, min(f, n // 2))
    if f <= 1: 
        return y
    env_in = np.linspace(0, 1, f, dtype=np.float32)
    env_out = env_in[::-1]
    y[:f] *= env_in
    y[-f:] *= env_out
    return y

def equal_power_xfade(a_tail, b_head):
    N = min(len(a_tail), len(b_head))
    if N == 0:
        return a_tail, 0
    t = np.linspace(0, 1, N, dtype=np.float32)
    w_a = np.cos(0.5 * np.pi * t) ** 2
    w_b = np.sin(0.5 * np.pi * t) ** 2
    mixed = a_tail[-N:] * w_a + b_head[:N] * w_b
    return mixed, N

def loop_to_duration_xfade(y, sr, target_sec, xfade_ms=150, fade_ms=10):
    target_samples = int(round(target_sec * sr))
    if len(y) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    xfade = int(sr * xfade_ms / 1000.0)
    if xfade >= len(y):
        xfade = max(1, len(y) // 4)

    out = y.astype(np.float32).copy()
    while len(out) < target_samples:
        if xfade > 0 and len(out) >= xfade:
            mixed, N = equal_power_xfade(out[-xfade:], y[:xfade])
            out = np.concatenate([out[:-xfade], mixed, y[xfade:]]).astype(np.float32)
        else:
            out = np.concatenate([out, y]).astype(np.float32)

    out = out[:target_samples]
    out = tiny_fade(out, sr, fade_ms=fade_ms)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:
        out = 0.999 * out / peak
    return out

def pad_to_duration_silence(y, sr, target_sec, fade_ms=10):
    target_samples = int(round(target_sec * sr))
    if len(y) >= target_samples:
        y = y[:target_samples]
    else:
        pad = np.zeros(target_samples - len(y), dtype=np.float32)
        y = np.concatenate([y, pad]).astype(np.float32)
    return tiny_fade(y, sr, fade_ms=fade_ms)

def write_collision_safe(dst_dir: Path, base_name: str, audio: np.ndarray, sr: int, subtype: str):
    dst = dst_dir / base_name
    if not dst.exists():
        sf.write(dst, audio, sr, subtype=subtype)
        return dst
    stem, suf = Path(base_name).stem, Path(base_name).suffix
    i = 1
    while True:
        alt = dst_dir / f"{stem}_{i}{suf}"
        if not alt.exists():
            sf.write(alt, audio, sr, subtype=subtype)
            return alt
        i += 1

def main():
    changed = 0
    skipped = 0
    # Only operate on top-level wavs; ignore review/aug folders
    for p in sorted(DATA_DIR.glob("*.wav")):
        if p.parent == AUG_DIR:
            continue
        if "review_flags" in str(p):
            continue

        y, sr = librosa.load(p, sr=None, mono=True)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        dur = len(y) / sr
        if dur >= TARGET_SEC - 1e-3:
            skipped += 1
            continue

        if MODE == "loop_xfade":
            y60 = loop_to_duration_xfade(y, sr, TARGET_SEC, xfade_ms=XFADE_MS, fade_ms=FADE_MS)
        elif MODE == "pad_silence":
            y60 = pad_to_duration_silence(y, sr, TARGET_SEC, fade_ms=FADE_MS)
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        # Save to AUG_DIR with same basename (easy to overwrite later after review)
        out_path = write_collision_safe(AUG_DIR, p.name, y60, sr, WRITE_SUBTYPE)
        changed += 1
        print(f"[AUG] {p.name}  ->  {out_path.name}")

    print(f"\nDone. Wrote {changed} augmented file(s) to {AUG_DIR}. Skipped {skipped} (already ≥ 60s).")

if __name__ == "__main__":
    main()

