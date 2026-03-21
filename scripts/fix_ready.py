# scripts/fix_water_ready.py
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import shutil

IN_DIR  = Path("data/processed_classes_filtered/water")
OUT_DIR = Path("data/processed_classes_filtered/water_ready")
TARGET_SR = 48000
TARGET_SEC = 60.0
DUR_TOL = 0.02  # +/- 20 ms
MODE = "loop_xfade"     # "loop_xfade" (recommended) or "pad_silence"
XFADE_MS = 150
FADE_MS = 10
PEAK_MAX = 0.999
GLOB = "*.wav"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def tiny_fade(y, sr, ms=10):
    n = len(y)
    f = int(sr * ms / 1000.0)
    f = max(1, min(f, n // 2))
    if f <= 1: return y
    env = np.linspace(0, 1, f, dtype=np.float32)
    y[:f] *= env
    y[-f:] *= env[::-1]
    return y

def equal_power_xfade(a_tail, b_head):
    N = min(len(a_tail), len(b_head))
    if N == 0: return a_tail, 0
    t = np.linspace(0, 1, N, dtype=np.float32)
    w_a = np.cos(0.5*np.pi*t)**2  # 1->0
    w_b = np.sin(0.5*np.pi*t)**2  # 0->1
    mixed = a_tail[-N:]*w_a + b_head[:N]*w_b
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
            mixed, _ = equal_power_xfade(out[-xfade:], y[:xfade])
            out = np.concatenate([out[:-xfade], mixed, y[xfade:]]).astype(np.float32)
        else:
            out = np.concatenate([out, y]).astype(np.float32)
    out = out[:target_samples]
    out = tiny_fade(out, sr, ms=fade_ms)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > PEAK_MAX:
        out = PEAK_MAX * out / peak
    return out

def pad_to_duration_silence(y, sr, target_sec, fade_ms=10):
    target_samples = int(round(target_sec * sr))
    if len(y) >= target_samples:
        y = y[:target_samples]
    else:
        pad = np.zeros(target_samples - len(y), dtype=np.float32)
        y = np.concatenate([y, pad]).astype(np.float32)
    return tiny_fade(y, sr, ms=fade_ms)

def fix_one(path: Path):
    # Load with target SR + mono to fix wrong_sr in one go
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    dur = len(y) / sr

    # Duration fixes
    if dur < TARGET_SEC - DUR_TOL:
        if MODE == "loop_xfade":
            y = loop_to_duration_xfade(y, sr, TARGET_SEC, xfade_ms=XFADE_MS, fade_ms=FADE_MS)
        else:
            y = pad_to_duration_silence(y, sr, TARGET_SEC, fade_ms=FADE_MS)
    elif dur > TARGET_SEC + DUR_TOL:
        y = y[:int(round(TARGET_SEC * sr))]
        y = tiny_fade(y, sr, ms=FADE_MS)
    else:
        # just tidy edges
        y = tiny_fade(y, sr, ms=FADE_MS)

    # Peak safety
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > PEAK_MAX:
        y = PEAK_MAX * y / peak

    # Write to OUT_DIR with same name
    out_path = OUT_DIR / path.name
    sf.write(out_path, y.astype(np.float32), sr, subtype="PCM_16")
    return out_path

def main():
    fixed, copied, skipped = 0, 0, 0
    for p in sorted(IN_DIR.glob(GLOB)):
        # only operate on top-level wavs; ignore review/aug folders if present
        if any(s in p.parts for s in ("review_flags", "augmented_60s", "backups_short60", "water_ready")):
            continue
        # Quick header read
        info = sf.info(str(p))
        sr_ok  = (info.samplerate == TARGET_SR)
        ch_ok  = (info.channels == 1)
        dur_ok = abs((info.frames / info.samplerate) - TARGET_SEC) <= DUR_TOL

        if sr_ok and ch_ok and dur_ok:
            # already ready → copy through for a complete ready set
            shutil.copy2(p, OUT_DIR / p.name)
            copied += 1
        else:
            outp = fix_one(p)
            fixed += 1
            if fixed <= 8:
                print(f"[FIX] {p.name} -> {outp.name}")

    total = fixed + copied
    print(f"\nDone. Ready files in {OUT_DIR}")
    print(f"Fixed: {fixed} | Copied (already ready): {copied} | Total written: {total}")

if __name__ == "__main__":
    main()

