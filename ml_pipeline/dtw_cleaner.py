import os
import sys
import numpy as np
import pandas as pd
from dtaidistance import dtw

# ── Path fix ──────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
# ──────────────────────────────────────────────────────────────────────────────

def clean_with_dtw(df):
    """Remove anomalous OD time series using DTW-based outlier detection"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    od_pairs = df.groupby(['origin_zone', 'dest_zone'])
    cleaned_pairs = []

    total = len(od_pairs)
    print(f"Processing {total} OD pairs...")

    for idx, ((orig, dest), group) in enumerate(od_pairs):
        series = group.sort_values('timestamp')['od_flow'].values.astype(float)
        if len(series) < 10:
            cleaned_pairs.append(group)
            continue

        if len(series) >= 48:
            template = np.median(series.reshape(-1, 48), axis=0)
            distances = []
            for i in range(0, len(series) - 47, 48):
                window = series[i:i + 48]
                if len(window) == 48:
                    d = dtw.distance_fast(
                        window.astype(np.double),
                        template.astype(np.double)
                    )
                    distances.append(d)

            if distances:
                threshold = np.percentile(distances, 95)
                valid_days = [i for i, d in enumerate(distances) if d <= threshold]
                valid_idx = []
                for v in valid_days:
                    valid_idx.extend(range(v * 48, v * 48 + 48))
                valid_idx = [i for i in valid_idx if i < len(series)]
                cleaned = group.iloc[valid_idx]
                cleaned_pairs.append(cleaned)
            else:
                cleaned_pairs.append(group)
        else:
            cleaned_pairs.append(group)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total} pairs...")

    return pd.concat(cleaned_pairs).reset_index(drop=True)


if __name__ == "__main__":
    input_path = os.path.join(DATASET_DIR, "od_signaling_data.csv")
    output_path = os.path.join(DATASET_DIR, "od_signaling_data.csv")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("Please run generate_dataset.py first from the project root.")
        sys.exit(1)

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records. Starting DTW cleaning...")

    df_clean = clean_with_dtw(df)

    df_clean.to_csv(output_path, index=False)
    print(f"\nDone! Cleaned dataset saved to: {output_path}")
    print(f"Records: {len(df)} → {len(df_clean)} (removed {len(df) - len(df_clean)} anomalous rows)")
