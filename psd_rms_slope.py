from pathlib import Path

import numpy as np
import pandas as pd

from hurst_psd_analysis import (
    CSV_FILES,
    X_COLUMN,
    PSD_COLUMN,
    GROUP_COLUMN,
    Q_SCALE,
    PSD_SCALE,
    RQ_INTEGRATION_Q_MAX,
    compute_rms_slope_from_psd,
)


def main() -> None:
    if not CSV_FILES:
        print("No CSV files specified in CSV_FILES in hurst_psd_analysis.py.")
        return

    for file_str in CSV_FILES:
        csv_path = Path(file_str)
        if not csv_path.is_file():
            print(f"WARNING: {csv_path} does not exist, skipping.")
            continue

        df = pd.read_csv(csv_path, sep=";")

        if X_COLUMN not in df.columns or PSD_COLUMN not in df.columns:
            print(
                f"{csv_path}: expected columns '{X_COLUMN}' and '{PSD_COLUMN}', "
                f"but found {list(df.columns)}"
            )
            continue

        if GROUP_COLUMN is None or GROUP_COLUMN not in df.columns:
            print(
                f"{csv_path}: GROUP_COLUMN '{GROUP_COLUMN}' is not present; "
                "treating whole file as one series."
            )
            groups = [("all", df)]
        else:
            groups = df.groupby(GROUP_COLUMN)

        print(f"\nRMS slope from PSD for {csv_path}:")
        print(f"(Integration up to q = {RQ_INTEGRATION_Q_MAX} nm^-1)")
        print(f"{'Series':30s}  {'m_rms (dimensionless)':>24s}")
        print("-" * 58)

        m_vals = []
        for series_name, subdf in groups:
            q = subdf[X_COLUMN].to_numpy() * Q_SCALE
            psd = subdf[PSD_COLUMN].to_numpy() * PSD_SCALE
            m_rms = compute_rms_slope_from_psd(q, psd)
            m_vals.append(m_rms)
            print(f"{str(series_name):30s}  {m_rms:24.4f}")

        if m_vals:
            m_arr = np.asarray(m_vals, dtype=float)
            m_mean = float(m_arr.mean())
            m_std = float(m_arr.std(ddof=1)) if m_arr.size > 1 else 0.0
            print("-" * 58)
            print(
                f"{'mean':30s}  {m_mean:24.4f}  (std = {m_std:.4f})"
            )


if __name__ == "__main__":
    main()

