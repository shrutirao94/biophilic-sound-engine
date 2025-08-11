import argparse
from pathlib import Path
import pandas as pd
import shutil

DATA_DIR = Path("data/processed_classes_filtered/water")
CSV_PATH = DATA_DIR / "quality_metrics_water.csv"
REVIEW_DIR = DATA_DIR / "review_flags"

def resolve_local_path(pstr: str) -> Path | None:
    p = Path(pstr)
    if p.exists():
        return p
    cand = DATA_DIR / Path(pstr).name
    return cand if cand.exists() else None

def safe_move(src: Path, dst_dir: Path, do_move: bool) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not do_move:
        return dst
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    # collision-safe rename
    stem, suf = src.stem, src.suffix
    i = 1
    while True:
        alt = dst_dir / f"{stem}_{i}{suf}"
        if not alt.exists():
            shutil.move(str(src), str(alt))
            return alt
        i += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Preview only; do not move files")
    args = ap.parse_args()
    DO_MOVE = not args.dry_run

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Metrics CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "file" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "file"})

    # thresholds (same as we agreed; derived from your dataset)
    q05 = df.quantile(0.05, numeric_only=True)
    q95 = df.quantile(0.95, numeric_only=True)
    DURATION_MIN = 5.0
    RMS_MIN = max(0.008, float(q05.get("rms", 0.0)))
    RMS_MAX = 0.25
    CLIPPED_RATIO_MAX = 0.001
    CREST_MAX = max(55.0, float(q95.get("crest_factor", 55.0)))
    ZCR_MAX = float(q95.get("zcr_mean", 0.2))
    FLATNESS_MAX = max(0.6, float(q95.get("spec_flatness_mean", 0.6)))
    AIR_RATIO_MAX = 0.70
    SILENCE_RATIO_MAX = 0.30
    ONSETS_MAX = float(df["onsets_per_sec"].quantile(0.98))
    FLUX_MAX = float(df["flux_mean"].quantile(0.98))
    STATIONARITY_MAX = float(df["stationarity"].quantile(0.98))
    DC_OFFSET_ABS_MAX = 0.01

    print("Mode:", "APPLY (moving files)" if DO_MOVE else "DRY RUN (preview only)")

    reasons_list = [
        "duration_short","rms_low","rms_high","clipped","crest_high","zcr_high",
        "flatness_high","air_hiss","silence_high","onsets_high","flux_high",
        "stationarity_high","dc_offset"
    ]
    for r in reasons_list:
        (REVIEW_DIR / r).mkdir(parents=True, exist_ok=True)
    MULTI_DIR = REVIEW_DIR / "multiple"
    MULTI_DIR.mkdir(parents=True, exist_ok=True)

    def reasons_for_row(row):
        r = []
        dur = float(row.get("duration_sec", 0.0))
        rms = float(row.get("rms", 0.0))
        clipped = float(row.get("clipped_ratio", 0.0))
        crest = float(row.get("crest_factor", 0.0))
        zcr = float(row.get("zcr_mean", 0.0))
        flat = float(row.get("spec_flatness_mean", 0.0))
        air = float(row.get("ratio_air_>8k", 0.0))
        sil = float(row.get("silence_ratio_-40dB", 0.0))
        onsets = float(row.get("onsets_per_sec", 0.0))
        flux = float(row.get("flux_mean", 0.0))
        stat = float(row.get("stationarity", 0.0))
        dc = float(row.get("dc_offset", 0.0))

        if dur < DURATION_MIN: r.append("duration_short")
        if rms < RMS_MIN: r.append("rms_low")
        if rms > RMS_MAX: r.append("rms_high")
        if clipped > CLIPPED_RATIO_MAX: r.append("clipped")
        if crest > CREST_MAX: r.append("crest_high")
        if zcr > ZCR_MAX: r.append("zcr_high")
        if flat > FLATNESS_MAX: r.append("flatness_high")
        if air > AIR_RATIO_MAX: r.append("air_hiss")
        if sil > SILENCE_RATIO_MAX: r.append("silence_high")
        if onsets > ONSETS_MAX: r.append("onsets_high")
        if flux > FLUX_MAX: r.append("flux_high")
        if stat > STATIONARITY_MAX: r.append("stationarity_high")
        if abs(dc) > DC_OFFSET_ABS_MAX: r.append("dc_offset")
        return r

    flag_counts = {k: 0 for k in reasons_list}
    multi_count = 0
    flagged_total = 0
    kept_total = 0
    missing = 0
    verbose = 0

    for _, row in df.iterrows():
        reasons = reasons_for_row(row)
        src = resolve_local_path(str(row.get("file", "")))

        if not reasons:
            kept_total += 1
            continue

        flagged_total += 1
        if src is None or not src.exists():
            missing += 1
            if verbose < 5:
                print(f"[MISS] {row.get('file','<none>')} -> cannot resolve in {DATA_DIR}")
                verbose += 1
            continue

        dst_dir = MULTI_DIR if len(reasons) > 1 else (REVIEW_DIR / reasons[0])
        if len(reasons) > 1:
            multi_count += 1
        else:
            flag_counts[reasons[0]] += 1

        if verbose < 8:
            print(f"[MOVE] {src} -> {dst_dir / src.name}")
            verbose += 1

        safe_move(src, dst_dir, DO_MOVE)

    print("\n=== Summary ===")
    print(f"Flagged: {flagged_total} | Kept: {kept_total} | Missing: {missing}")
    print("By reason:")
    for k, v in flag_counts.items():
        print(f"  {k:18s} {v}")
    print(f"  {'multiple':18s} {multi_count}")
    print(f"\nReview root: {REVIEW_DIR.resolve()}")

if __name__ == "__main__":
    main()

