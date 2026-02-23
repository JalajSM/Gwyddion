import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==== USER-CONFIGURABLE SETTINGS ============================================

# List of CSV files to analyse.
# You can use absolute paths or paths relative to the folder where you run this script.
CSV_FILES: List[str] = [
    # PSD exported from TopoBank for 1D profile in x:
    r"modules/topobank_statisticspower_spectral_density.csv",
]

# Column names in the CSV for wavevector (x-axis) and PSD (y-axis).
# Set these to the exact header strings from your CSV file.
X_COLUMN: str = "Wavevector (1/m)"
PSD_COLUMN: str = "PSD (m³)"

# Optional grouping column: if present, the script will treat each unique
# value as a separate series and fit an H value for each.
GROUP_COLUMN: Optional[str] = "Subject name"

# Unit scaling:
#   - Input wavevector is in 1/m, but we want to plot in nm^-1.
#   - Input PSD is in m^3, but we want to plot in nm^3.
Q_SCALE: float = 1e-9   # 1/m -> 1/nm
PSD_SCALE: float = 1e27  # m^3 -> nm^3

# Fitting range for the wavevector, expressed in nm^-1 after scaling.
# Set to None to use the full available range.
FIT_RANGE: Optional[Tuple[float, float]] = (8e-3, 2e-1)

# Upper limit for RMS roughness integration in nm^-1
RQ_INTEGRATION_Q_MAX: float = 0.2

# Directory where output plots will be saved (created if it does not exist).
OUTPUT_DIR: Path = Path("output_hurst_plots")


# ==== CORE COMPUTATION FUNCTIONS ===========================================

def fit_loglog_1d_psd(
    q: np.ndarray,
    psd: np.ndarray,
    fit_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Fit log10(PSD) vs log10(q) with a straight line.

    Returns
    -------
    slope : float
        Slope of log10(PSD) vs log10(q).
    intercept : float
        Intercept of the fit.
    """
    q = np.asarray(q, dtype=float)
    psd = np.asarray(psd, dtype=float)

    # Valid data: finite and positive PSD (required for logarithm)
    mask = np.isfinite(q) & np.isfinite(psd) & (psd > 0)

    if fit_range is not None:
        qmin, qmax = fit_range
        mask &= (q >= qmin) & (q <= qmax)

    q_sel = q[mask]
    psd_sel = psd[mask]

    if q_sel.size < 2:
        raise ValueError("Not enough points in the selected fit range.")

    logq = np.log10(q_sel)
    logpsd = np.log10(psd_sel)

    slope, intercept = np.polyfit(logq, logpsd, 1)
    return slope, intercept


def slope_to_hurst_1d(slope: float) -> float:
    """
    Convert log–log PSD slope to H for a 1D profile.

    For a self-affine 1D profile:
        PSD(q) ~ q^{-(2H + 1)}
      => log10(PSD) = -(2H + 1) * log10(q) + const
      => slope = -(2H + 1)
      => H = (-slope - 1) / 2
    """
    return (-slope - 1.0) / 2.0


def compute_rq_from_psd(
    q: np.ndarray,
    psd: np.ndarray,
    q_max: float = RQ_INTEGRATION_Q_MAX,
) -> float:
    """
    Compute RMS roughness R_q from a 1D PSD using

        R_q^2 = (1 / pi) * ∫ C(q) dq   over  0 <= q <= q_max

    and return R_q = sqrt(R_q^2).

    Parameters
    ----------
    q : array-like
        Wavevector values in nm^-1.
    psd : array-like
        PSD values in nm^3.
    q_max : float
        Upper integration limit in nm^-1.
    """
    q = np.asarray(q, dtype=float)
    psd = np.asarray(psd, dtype=float)

    mask = np.isfinite(q) & np.isfinite(psd) & (psd >= 0.0) & (q <= q_max)
    q_sel = q[mask]
    psd_sel = psd[mask]

    if q_sel.size < 2:
        return float("nan")

    integral = np.trapz(psd_sel, q_sel)

    # Allow for possible sign issues from unsorted q by taking absolute value.
    rq_sq = abs((1.0 / math.pi) * integral)
    return float(math.sqrt(rq_sq))


def compute_rms_slope_from_psd(
    q: np.ndarray,
    psd: np.ndarray,
    q_max: float = RQ_INTEGRATION_Q_MAX,
) -> float:
    """
    Compute RMS slope m_rms from a 1D PSD using

        m_rms^2 = (1 / pi) * ∫ q^2 C(q) dq   over  0 <= q <= q_max

    and return m_rms = sqrt(m_rms^2).

    Parameters
    ----------
    q : array-like
        Wavevector values in nm^-1.
    psd : array-like
        PSD values in nm^3.
    q_max : float
        Upper integration limit in nm^-1.
    """
    q = np.asarray(q, dtype=float)
    psd = np.asarray(psd, dtype=float)

    mask = np.isfinite(q) & np.isfinite(psd) & (psd >= 0.0) & (q <= q_max)
    q_sel = q[mask]
    psd_sel = psd[mask]

    if q_sel.size < 2:
        return float("nan")

    integrand = (q_sel ** 2) * psd_sel
    integral = np.trapz(integrand, q_sel)

    m_sq = abs((1.0 / math.pi) * integral)
    return float(math.sqrt(m_sq))


# ==== PLOTTING =============================================================

def plot_psd_with_fit(
    q: np.ndarray,
    psd: np.ndarray,
    slope: float,
    intercept: float,
    H: float,
    csv_path: Path,
    fit_range: Optional[Tuple[float, float]] = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Create a log–log PSD plot with a single fitted line and equation annotated.

    Returns
    -------
    output_path : Path
        Path to the saved image file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    q = np.asarray(q, dtype=float)
    psd = np.asarray(psd, dtype=float)

    # Prepare fit line in the fitted q-range
    mask_fit = np.isfinite(q) & np.isfinite(psd) & (psd > 0)
    if fit_range is not None:
        qmin, qmax = fit_range
        mask_fit &= (q >= qmin) & (q <= qmax)

    q_fit = q[mask_fit]
    if q_fit.size >= 2:
        q_fit_sorted = np.sort(q_fit)
        logq_fit = np.log10(q_fit_sorted)
        logpsd_fit = slope * logq_fit + intercept
        psd_fit = 10.0 ** logpsd_fit
    else:
        q_fit_sorted = np.array([])
        psd_fit = np.array([])

    fig, ax = plt.subplots(figsize=(7, 4))

    # Scatter of original data
    ax.scatter(q, psd, s=25, alpha=0.7, label="PSD data")

    # Fitted line
    if q_fit_sorted.size > 0:
        ax.plot(
            q_fit_sorted,
            psd_fit,
            color="red",
            linewidth=2,
            label="Linear fit (log–log)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavevector (nm$^{-1}$)")
    ax.set_ylabel("PSD (nm$^{3}$)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Annotate equation and H on the plot (in log10 space)
    eq_text = (
        r"$\log_{10}(\mathrm{PSD}) = "
        f"{slope:.3f} \\, \log_{10}(q) + {intercept:.3f}$\n"
        r"$H = \frac{{-m - 1}}{2} = " + f"{H:.3f}$"
    )

    ax.text(
        0.03,
        0.97,
        eq_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Title = CSV file name
    ax.set_title(csv_path.name)
    ax.legend()

    # Save figure
    base_name = csv_path.stem + "_hurst_fit.png"
    output_path = output_dir / base_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def plot_psd_with_fit_multi(
    groups: List[dict],
    csv_path: Path,
    fit_range: Optional[Tuple[float, float]] = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Create a log–log PSD plot for multiple series, each with its own fit.

    Each element of ``groups`` must be a dict with keys:
        - "name": label for the series
        - "q": 1D array of wavevectors
        - "psd": 1D array of PSDs
        - "slope": fitted slope
        - "intercept": fitted intercept
        - "H": Hurst exponent

    A table of H values is added to the figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(groups), 2)))

    for idx, (g, color) in enumerate(zip(groups, colors)):
        q = np.asarray(g["q"], dtype=float)
        psd = np.asarray(g["psd"], dtype=float)
        slope = float(g["slope"])
        intercept = float(g["intercept"])

        # Scatter of original data
        ax.scatter(q, psd, s=15, alpha=0.6, label=g["name"], color=color)

        # Prepare fit line
        mask_fit = np.isfinite(q) & np.isfinite(psd) & (psd > 0)
        if fit_range is not None:
            qmin, qmax = fit_range
            mask_fit &= (q >= qmin) & (q <= qmax)

        q_fit = q[mask_fit]
        if q_fit.size >= 2:
            q_fit_sorted = np.sort(q_fit)
            logq_fit = np.log10(q_fit_sorted)
            logpsd_fit = slope * logq_fit + intercept
            psd_fit = 10.0 ** logpsd_fit
            ax.plot(
                q_fit_sorted,
                psd_fit,
                color=color,
                linewidth=2,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavevector (nm$^{-1}$)")
    ax.set_ylabel("PSD (nm$^{3}$)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_title(csv_path.name)
    ax.legend(loc="upper right", fontsize=8)

    # Build table of H, slopes, Rq, and m_rms (for separate table figure)
    table_rows = []
    H_values = []
    Rq_values = []
    m_values = []
    for g in groups:
        table_rows.append(
            [
                str(g["name"]),
                f"{g['H']:.4f}",
                f"{g['slope']:.4f}",
                f"{g['Rq']:.4f}",
                f"{g['m_rms']:.4f}",
            ]
        )
        H_values.append(float(g["H"]))
        Rq_values.append(float(g["Rq"]))
        m_values.append(float(g["m_rms"]))

    H_mean = float(np.mean(H_values)) if H_values else float("nan")
    Rq_mean = float(np.mean(Rq_values)) if Rq_values else float("nan")
    m_mean = float(np.mean(m_values)) if m_values else float("nan")

    # Note mean(H), mean(Rq), and mean(m_rms) on the main plot
    ax.text(
        0.03,
        0.97,
        "mean(H) = "
        f"{H_mean:.4f}\nmean(Rq) = {Rq_mean:.4f} nm\nmean(m_rms) = {m_mean:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Save main plot without table overlay
    base_name = csv_path.stem + "_hurst_multi_fit.png"
    plot_path = output_dir / base_name
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    # Create a separate figure for the table
    fig_table, ax_table = plt.subplots(figsize=(7.5, 0.6 + 0.35 * len(table_rows)))
    ax_table.axis("off")

    the_table = ax_table.table(
        cellText=table_rows,
        colLabels=["Series", "H", "slope", "Rq (nm)", "m_rms"],
        loc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.0, 1.2)

    fig_table.tight_layout()

    table_name = csv_path.stem + "_hurst_multi_table.png"
    table_path = output_dir / table_name
    fig_table.savefig(table_path, dpi=300)
    plt.close(fig_table)

    return plot_path


# ==== HIGH-LEVEL ANALYSIS PIPELINE =========================================

def analyse_csv_file(
    csv_path: Path,
    x_column: str,
    psd_column: str,
    fit_range: Optional[Tuple[float, float]],
    output_dir: Path,
) -> Tuple[float, float, float]:
    """
    Load a single CSV, fit slope(s), compute H value(s), make plot(s), and save.

    If GROUP_COLUMN is present, a separate fit is performed for each series and
    a combined plot with a table of H is saved. The function then returns the
    mean slope and H across all series.

    Returns
    -------
    slope : float
    intercept : float
        For grouped data, this is the mean intercept across series.
    H : float
        Mean H across all series in this file.
    """
    # TopoBank exports this PSD file with semicolon separators.
    df = pd.read_csv(csv_path, sep=";")

    if x_column not in df.columns or psd_column not in df.columns:
        raise KeyError(
            f"{csv_path}: expected columns '{x_column}' and '{psd_column}'. "
            f"Available columns are: {list(df.columns)}"
        )

    # If a grouping column is defined and present, do per-series fits
    if GROUP_COLUMN is not None and GROUP_COLUMN in df.columns:
        groups_info: List[dict] = []
        for series_name, subdf in df.groupby(GROUP_COLUMN):
            # Convert input units:
            #   Wavevector: 1/m -> 1/nm
            #   PSD:       m^3 -> nm^3
            q = subdf[x_column].to_numpy() * Q_SCALE
            psd = subdf[psd_column].to_numpy() * PSD_SCALE

            slope, intercept = fit_loglog_1d_psd(q, psd, fit_range=fit_range)
            H = slope_to_hurst_1d(slope)
            Rq = compute_rq_from_psd(q, psd)
            m_rms = compute_rms_slope_from_psd(q, psd)

            groups_info.append(
                {
                    "name": series_name,
                    "q": q,
                    "psd": psd,
                    "slope": slope,
                    "intercept": intercept,
                    "H": H,
                    "Rq": Rq,
                    "m_rms": m_rms,
                }
            )

        if not groups_info:
            raise ValueError(
                f"{csv_path}: no data found after grouping by '{GROUP_COLUMN}'."
            )

        img_path = plot_psd_with_fit_multi(
            groups=groups_info,
            csv_path=csv_path,
            fit_range=fit_range,
            output_dir=output_dir,
        )

        print("Per-series results:")
        for g in groups_info:
            print(
                f"  {csv_path} | {g['name']}: "
                f"slope = {g['slope']:.4f}, H = {g['H']:.4f}, "
                f"Rq = {g['Rq']:.4f} nm, m_rms = {g['m_rms']:.4f}"
            )

        slopes = np.array([g["slope"] for g in groups_info], dtype=float)
        intercepts = np.array([g["intercept"] for g in groups_info], dtype=float)
        H_values = np.array([g["H"] for g in groups_info], dtype=float)
        Rq_values = np.array([g["Rq"] for g in groups_info], dtype=float)
        m_values = np.array([g["m_rms"] for g in groups_info], dtype=float)

        slope_mean = float(slopes.mean())
        intercept_mean = float(intercepts.mean())
        H_mean = float(H_values.mean())
        H_std = float(H_values.std(ddof=1)) if H_values.size > 1 else 0.0
        Rq_mean = float(Rq_values.mean())
        Rq_std = float(Rq_values.std(ddof=1)) if Rq_values.size > 1 else 0.0
        m_mean = float(m_values.mean())
        m_std = float(m_values.std(ddof=1)) if m_values.size > 1 else 0.0

        print(
            f"\n{csv_path} summary over {len(groups_info)} series: "
            f"mean(H) = {H_mean:.4f}, std(H) = {H_std:.4f}; "
            f"mean(Rq) = {Rq_mean:.4f} nm, std(Rq) = {Rq_std:.4f} nm; "
            f"mean(m_rms) = {m_mean:.4f}, std(m_rms) = {m_std:.4f}"
        )

        # Save per-series statistics to CSV
        stats_rows = []
        for g in groups_info:
            stats_rows.append(
                {
                    "Series": g["name"],
                    "slope": g["slope"],
                    "H": g["H"],
                    "Rq_nm": g["Rq"],
                    "m_rms": g["m_rms"],
                }
            )
        stats_rows.append(
            {
                "Series": "mean",
                "slope": slope_mean,
                "H": H_mean,
                "Rq_nm": Rq_mean,
                "m_rms": m_mean,
            }
        )

        stats_df = pd.DataFrame(stats_rows)
        stats_path = output_dir / f"{csv_path.stem}_hurst_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        print(f"Per-series statistics saved to: {stats_path}")
        print(f"Combined plot with table saved to: {img_path}")

        return slope_mean, intercept_mean, H_mean

    # Fallback: treat the whole file as one series, with unit conversion
    q = df[x_column].to_numpy() * Q_SCALE
    psd = df[psd_column].to_numpy() * PSD_SCALE

    slope, intercept = fit_loglog_1d_psd(q, psd, fit_range=fit_range)
    H = slope_to_hurst_1d(slope)

    img_path = plot_psd_with_fit(
        q=q,
        psd=psd,
        slope=slope,
        intercept=intercept,
        H=H,
        csv_path=csv_path,
        fit_range=fit_range,
        output_dir=output_dir,
    )

    print(
        f"{csv_path} -> slope = {slope:.4f}, H = {H:.4f}, "
        f"plot saved to: {img_path}"
    )

    return slope, intercept, H


def main() -> None:
    if not CSV_FILES:
        print("No CSV files specified in CSV_FILES. Edit the script to add them.")
        return

    slopes: List[float] = []
    H_values: List[float] = []

    for file_str in CSV_FILES:
        csv_path = Path(file_str)
        if not csv_path.is_file():
            print(f"WARNING: {csv_path} does not exist, skipping.")
            continue

        slope, intercept, H = analyse_csv_file(
            csv_path=csv_path,
            x_column=X_COLUMN,
            psd_column=PSD_COLUMN,
            fit_range=FIT_RANGE,
            output_dir=OUTPUT_DIR,
        )
        slopes.append(slope)
        H_values.append(H)

    if H_values:
        H_arr = np.asarray(H_values)
        H_mean = float(H_arr.mean())
        H_std = float(H_arr.std(ddof=1)) if H_arr.size > 1 else 0.0
        print(
            f"\nSummary over {len(H_values)} file(s): "
            f"mean(H) = {H_mean:.4f}, std(H) = {H_std:.4f}"
        )
    else:
        print("No valid results were computed.")


if __name__ == "__main__":
    main()

