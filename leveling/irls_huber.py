import json
import sqlite3
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import kurtosis, skew, median_abs_deviation
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def robust_polynomial_fit(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    max_iter: int = 50,
    tol: float = 1e-6,
    weight_fun: str = "huber",  # 'huber' or 'tukey'
    c: float = 1.345,  # huber_c or tukey_c (behavior depends on weight_fun)
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IRLS robust polynomial fit.
    Returns:
        coef (np.ndarray): polynomial coefficients in increasing order [c0, c1, ..., c_deg]
        weights (np.ndarray): final weights for each sample
    Notes:
        - Uses MAD scale converted to sigma: s = 1.4826 * MAD
        - X design uses increasing powers: [1, x, x^2, ...] matching Polynomial([...])
    """
    if len(x) < deg + 1:
        # not enough points, return constant fit
        coef0 = np.zeros(deg + 1)
        coef0[0] = np.mean(y)
        return coef0, np.ones_like(y)

    # design matrix (columns: 1, x, x^2, ...)
    X = np.vander(x, N=deg + 1, increasing=True)
    n = len(y)
    weights = np.ones(n, dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]

    for it in range(1, max_iter + 1):
        W_sqrt = np.sqrt(weights)
        # weighted least squares
        Xw = X * W_sqrt[:, None]
        yw = y * W_sqrt
        try:
            coef_new, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            if debug:
                print(f"IRLS: LinAlgError at iter {it}")
            break

        y_pred = X @ coef_new
        residuals = y - y_pred

        # robust scale: convert MAD -> sigma for normal approx
        # scipy returns scaled if scale='normal'
        mad_val = median_abs_deviation(residuals, scale='normal')
        # median_abs_deviation with scale='normal' returns 1.4826*MAD already
        s = mad_val if mad_val > 0 else np.std(residuals, ddof=1)
        if s == 0:
            weights = np.ones_like(residuals)
            coef = coef_new
            break

        u = residuals / s  # standardized residuals

        if weight_fun.lower() == "huber":
            # Huber weights
            w = np.ones_like(u)
            mask = np.abs(u) > c
            w[mask] = c / np.abs(u[mask])
        elif weight_fun.lower() == "tukey":
            # Tukey biweight (redescending)
            w = np.zeros_like(u)
            mask = np.abs(u) <= c
            # Tukey: (1 - (u/c)^2)^2
            w[mask] = (1.0 - (u[mask] / c) ** 2) ** 2
        else:
            raise ValueError("weight_fun must be 'huber' or 'tukey'")

        # convergence check on coefficients
        if np.linalg.norm(coef_new - coef) < tol:
            coef = coef_new
            weights = w
            if debug:
                print(f"IRLS converged at iter {it}")
            break

        coef = coef_new
        weights = w
        if debug:
            print(
                f"IRLS iter {it}: scale={s:.4f}, coef_norm={np.linalg.norm(coef):.4f}, w_min={weights.min():.4f}")

    # ensure coef is increasing order for Polynomial([...])
    return coef, weights


def two_stage_tukey_huber_fit(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    tukey_c: float = 3.5,
    huber_c: float = 1.0,
    tukey_outlier_thresh: float = 0.05,
    debug: bool = False,
) -> Tuple[Polynomial, np.ndarray, Dict[str, Any]]:
    """
    Two-stage: 1) Tukey (strong) to downweight/identify extreme outliers
               2) Huber on the remaining (or weighted) points to finalize fit
    Returns:
        Polynomial object (trend), final_weights (on full sample), qc_info dict
    """
    # Stage 1: Tukey IRLS on all points
    coef1, w1 = robust_polynomial_fit(
        x, y, deg=deg, weight_fun='tukey', c=tukey_c, debug=debug)
    # Mark extreme-outliers as those with very small tukey weights
    outlier_mask = w1 < tukey_outlier_thresh

    # Stage 2: Huber IRLS on inliers (or keep weights but re-fit)
    if outlier_mask.all():
        # If everything is marked as outlier (pathological), fallback to Huber on all
        coef2, w2 = robust_polynomial_fit(
            x, y, deg=deg, weight_fun='huber', c=huber_c, debug=debug)
        final_weights = w2
    else:
        # Option A: remove strong outliers and fit Huber on remaining
        x_in = x[~outlier_mask]
        y_in = y[~outlier_mask]
        coef2_in, w2_in = robust_polynomial_fit(
            x_in, y_in, deg=deg, weight_fun='huber', c=huber_c, debug=debug)
        # Build final weights over full set: keep tukey small weights for outliers, use w2_in for inliers
        final_weights = np.zeros_like(y, dtype=float)
        final_weights[outlier_mask] = 0.0
        final_weights[~outlier_mask] = w2_in
        # For constructing Polynomial over full domain, we need coefficients over inliers padded to degree
        # We can use coef2_in as final coefficients (fitted on inliers)
        coef2 = coef2_in

    # QC info
    y_pred_before = np.polyval(
        # placeholder
        coef[::-1], x) if False else (np.vander(x, deg+1, increasing=True) @ coef)
    # Use final coef to compute final prediction
    trend_pred = np.vander(x, deg+1, increasing=True) @ coef2
    residuals_before = y - (np.vander(x, deg+1, increasing=True) @
                            np.linalg.lstsq(np.vander(x, deg+1, increasing=True), y, rcond=None)[0])
    residuals_after = y - trend_pred

    qc = {
        "n_points": len(x),
        "n_outliers_stage1": int(np.sum(outlier_mask)),
        "pct_outliers_stage1": float(np.mean(outlier_mask)),
        "rmse_before": float(np.sqrt(np.mean(residuals_before ** 2))),
        "rmse_after": float(np.sqrt(np.mean(residuals_after ** 2))),
        "mae_before": float(np.mean(np.abs(residuals_before))),
        "mae_after": float(np.mean(np.abs(residuals_after))),
        "kurtosis_before": float(kurtosis(residuals_before, fisher=False)),
        "kurtosis_after": float(kurtosis(residuals_after, fisher=False)),
        "skew_before": float(skew(residuals_before)),
        "skew_after": float(skew(residuals_after)),
    }

    # return Polynomial object constructed from coef2 (coef2 is increasing order)
    poly = Polynomial(coef2)
    return poly, final_weights, qc


def statistical_level_robust(
    db_path: str,
    table: str,
    line_id_col: str = "LINE",
    type_filter: str = "TIE",  # "TIE" or "LINE" or "SELECTED"
    input_ch: str = "TIE_LEVEL",
    diff_ch: str = "CROSS_DIFF",
    output_ch: str = "LEVELLED",
    trend_order: int = 1,
    trend_out_ch: str = "TREND",
    robust: bool = True,
    two_stage: bool = True,
    tukey_c: float = 3.5,
    huber_c: float = 1.0,
    tukey_outlier_thresh: float = 0.05,
    spatial_consistency: bool = True,
    spatial_radius: float = 50.0,
    spatial_alpha: float = 0.5,
    debug: bool = False,
) -> str:
    """
    Replace your original statistical_level with this robust version.
    Saves QC into a new SQLite table 'leveling_qc'.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)

    # checks
    for ch in [input_ch, diff_ch]:
        if ch not in df.columns:
            conn.close()
            raise ValueError(f"缺少通道：{ch}")

    # selected lines
    all_lines = df[line_id_col].unique()
    if type_filter in ["TIE", "LINE"]:
        selected_lines = [l for l in all_lines if str(
            l).startswith("T" if type_filter == "TIE" else "L")]
    else:
        selected_lines = list(all_lines)

    if output_ch not in df.columns:
        df[output_ch] = df[input_ch]

    if trend_out_ch and trend_out_ch not in df.columns:
        df[trend_out_ch] = np.nan

    # Prepare QC table
    qc_records = []

    # If spatial consistency requested, try to gather X,Y
    has_xy = ("X" in df.columns) and ("Y" in df.columns)
    if spatial_consistency and has_xy:
        coords = np.column_stack([df["X"].values, df["Y"].values])
        tree = cKDTree(coords)
    else:
        tree = None

    # iterate lines
    df_diff = df[[line_id_col, diff_ch]].dropna()
    corrections = {}

    for line in selected_lines:
        line_mask = df[line_id_col] == line
        line_diffs = df_diff[df_diff[line_id_col] == line]
        if line_diffs.empty:
            continue

        x_idx = line_diffs.index.values.astype(float)
        y_vals = line_diffs[diff_ch].values.astype(float)
        x_vals = x_idx.copy()  # index as X coordinate along line (same as before)

        if len(x_vals) < trend_order + 1:
            # can't fit
            continue

        # Fit using robust or OLS
        if robust:
            if two_stage:
                poly, final_weights, qc = two_stage_tukey_huber_fit(
                    x_vals, y_vals, deg=trend_order,
                    tukey_c=tukey_c, huber_c=huber_c,
                    tukey_outlier_thresh=tukey_outlier_thresh, debug=debug
                )
            else:
                coef_final, final_weights = robust_polynomial_fit(
                    x_vals, y_vals, deg=trend_order, weight_fun='huber', c=huber_c, debug=debug
                )
                poly = Polynomial(coef_final)
                # compute qc
                y_pred_before = np.vander(x_vals, trend_order+1, increasing=True) @ np.linalg.lstsq(
                    np.vander(x_vals, trend_order+1, increasing=True), y_vals, rcond=None)[0]
                residuals_after = y_vals - poly(x_vals)
                residuals_before = y_vals - y_pred_before
                qc = {
                    "n_points": len(x_vals),
                    "n_outliers_stage1": int(np.sum(final_weights < 1e-6)),
                    "pct_outliers_stage1": float(np.mean(final_weights < 1e-6)),
                    "rmse_before": float(np.sqrt(np.mean(residuals_before ** 2))),
                    "rmse_after": float(np.sqrt(np.mean(residuals_after ** 2))),
                    "mae_before": float(np.mean(np.abs(residuals_before))),
                    "mae_after": float(np.mean(np.abs(residuals_after))),
                    "kurtosis_before": float(kurtosis(residuals_before, fisher=False)),
                    "kurtosis_after": float(kurtosis(residuals_after, fisher=False)),
                    "skew_before": float(skew(residuals_before)),
                    "skew_after": float(skew(residuals_after)),
                }
        else:
            # OLS fallback
            coef = np.linalg.lstsq(
                np.vander(x_vals, trend_order+1, increasing=True), y_vals, rcond=None)[0]
            poly = Polynomial(coef)
            final_weights = np.ones_like(y_vals)
            residuals_before = y_vals - poly(x_vals)
            qc = {
                "n_points": len(x_vals),
                "n_outliers_stage1": 0,
                "pct_outliers_stage1": 0.0,
                "rmse_before": float(np.sqrt(np.mean(residuals_before ** 2))),
                "rmse_after": float(np.sqrt(np.mean(residuals_before ** 2))),
                "mae_before": float(np.mean(np.abs(residuals_before))),
                "mae_after": float(np.mean(np.abs(residuals_before))),
                "kurtosis_before": float(kurtosis(residuals_before, fisher=False)),
                "kurtosis_after": float(kurtosis(residuals_before, fisher=False)),
                "skew_before": float(skew(residuals_before)),
                "skew_after": float(skew(residuals_before)),
            }

        # Spatial consistency adjustment (optional): boost weight if neighbors share similar residual
        if spatial_consistency and tree is not None:
            # compute local_std for each point
            indices = line_diffs.index.to_numpy()
            local_std_list = []
            for i, idx in enumerate(indices):
                xi, yi = df.loc[idx, "X"], df.loc[idx, "Y"]
                nbrs = tree.query_ball_point([xi, yi], spatial_radius)
                if len(nbrs) <= 1:
                    local_std = 0.0
                else:
                    # use residuals from after-fit
                    resid = (df.loc[nbrs, diff_ch] -
                             poly(df.loc[nbrs].index.astype(float))).dropna()
                    local_std = float(np.std(resid)) if len(resid) > 0 else 0.0
                local_std_list.append(local_std)
            local_std = np.array(local_std_list)
            glob_std = np.std(y_vals) if np.std(y_vals) > 0 else 1.0
            spatial_factor = 1.0 / \
                (1.0 + spatial_alpha * (local_std / glob_std))
            # integrate spatial factor into final_weights (interpolate mapping: indices -> positions in final_weights)
            # final_weights currently aligned to x_vals order
            final_weights = final_weights * spatial_factor

        # Store correction function & apply to df
        corrections[line] = poly
        # apply to full line rows
        mask_line_all = df[line_id_col] == line
        idxs_all = df[mask_line_all].index.astype(float)
        trend_vals = poly(idxs_all)

        if type_filter == "LINE":
            df.loc[mask_line_all, output_ch] = df.loc[mask_line_all,
                                                      input_ch].values - trend_vals
        else:
            df.loc[mask_line_all, output_ch] = df.loc[mask_line_all,
                                                      input_ch].values + trend_vals

        if trend_out_ch:
            df.loc[mask_line_all, trend_out_ch] = trend_vals

        # finalize QC record for sqlite
        qc_record = {
            "line": str(line),
            "n_points": qc["n_points"],
            "n_outliers_stage1": qc["n_outliers_stage1"],
            "pct_outliers_stage1": qc["pct_outliers_stage1"],
            "rmse_before": qc["rmse_before"],
            "rmse_after": qc["rmse_after"],
            "mae_before": qc["mae_before"],
            "mae_after": qc["mae_after"],
            "kurtosis_before": qc["kurtosis_before"],
            "kurtosis_after": qc["kurtosis_after"],
            "skew_before": qc["skew_before"],
            "skew_after": qc["skew_after"],
            "weights_summary": json.dumps({
                "min": float(np.min(final_weights)),
                "median": float(np.median(final_weights)),
                "mean": float(np.mean(final_weights)),
                "max": float(np.max(final_weights)),
                "pct_zero": float(np.mean(final_weights <= 1e-6))
            }),
            "params": json.dumps({
                "robust": bool(robust),
                "two_stage": bool(two_stage),
                "tukey_c": float(tukey_c),
                "huber_c": float(huber_c),
                "trend_order": int(trend_order)
            })
        }
        qc_records.append(qc_record)

    # write back main table
    df.to_sql(table, conn, if_exists='replace', index=False)

    # write QC table (replace)
    qc_df = pd.DataFrame(qc_records)
    if not qc_df.empty:
        qc_df.to_sql("leveling_qc", conn, if_exists='replace', index=False)

    conn.close()
    return f"Robust statistical leveling complete for {type_filter} lines. Output: {output_ch}, Trend: {trend_out_ch}. QC table: leveling_qc"


def evaluate_on_table(db_path: str, table: str, line_id_col='LINE', diff_ch='CROSS_DIFF') -> Dict[str, Any]:
    """
    Quick aggregate evaluator: load table and return global stats on CROSS_DIFF
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"SELECT {diff_ch} FROM {table} WHERE {diff_ch} IS NOT NULL", conn)
    arr = df[diff_ch].values
    conn.close()
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mae": float(np.mean(np.abs(arr))),
        "max": float(np.max(np.abs(arr))),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "kurtosis": float(kurtosis(arr, fisher=False)),
        "skew": float(skew(arr))
    }


def grid_search_tukey_huber(
    db_path: str,
    table: str,
    line_id_col: str = "LINE",
    diff_ch: str = "CROSS_DIFF",
    degs: List[int] = [1, 2],
    tukey_cs: List[float] = [3.0, 3.5, 4.5],
    huber_cs: List[float] = [0.8, 1.0, 1.345],
    debug: bool = False,
    out_prefix: str = "grid_search_results"
) -> pd.DataFrame:
    """
    Run grid search over provided parameter sets.
    Produces:
      - CSV of aggregate results
      - simple plots (rmse vs kurtosis) saved to PNG files with out_prefix
    Returns results DataFrame.
    """
    # load full table (we'll perform per-line fits and aggregate)
    conn = sqlite3.connect(db_path)
    df_full = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()

    # select lines that have CROSS_DIFF
    df_diff = df_full[[line_id_col, diff_ch]].dropna()
    lines = df_diff[line_id_col].unique()

    results = []
    total_runs = len(degs) * len(tukey_cs) * len(huber_cs)
    run_idx = 0
    for deg in degs:
        for tc in tukey_cs:
            for hc in huber_cs:
                run_idx += 1
                if debug:
                    print(
                        f"Grid run {run_idx}/{total_runs}: deg={deg}, tukey_c={tc}, huber_c={hc}")
                # clone df for applying corrections in-memory
                df_work = df_full.copy()
                per_line_stats = []
                # apply line-by-line robust leveling non-destructively (only evaluate CROSS_DIFF residual stats)
                for line in lines:
                    line_mask = df_work[line_id_col] == line
                    line_diffs = df_diff[df_diff[line_id_col] == line]
                    if line_diffs.empty:
                        continue
                    x_idx = line_diffs.index.values.astype(float)
                    y_vals = line_diffs[diff_ch].values.astype(float)
                    if len(x_idx) < deg + 1:
                        continue
                    try:
                        poly, final_weights, qc = two_stage_tukey_huber_fit(
                            x_idx, y_vals, deg=deg, tukey_c=tc, huber_c=hc, debug=False)
                    except Exception as e:
                        if debug:
                            print(f"line {line} failed: {e}")
                        continue
                    # compute residuals after trend removal
                    residuals_after = y_vals - poly(x_idx)
                    per_line_stats.append({
                        "line": str(line),
                        "rmse_after": float(np.sqrt(np.mean(residuals_after ** 2))),
                        "mae_after": float(np.mean(np.abs(residuals_after))),
                        "kurtosis_after": float(kurtosis(residuals_after, fisher=False)),
                        "skew_after": float(skew(residuals_after)),
                        "n": len(y_vals),
                        "pct_outliers_stage1": float(np.mean(final_weights <= 1e-6))
                    })
                # aggregate
                if len(per_line_stats) == 0:
                    continue
                per_df = pd.DataFrame(per_line_stats)
                agg = {
                    "deg": deg,
                    "tukey_c": tc,
                    "huber_c": hc,
                    "lines_used": len(per_df),
                    "mean_rmse_after": float(per_df["rmse_after"].mean()),
                    "median_rmse_after": float(per_df["rmse_after"].median()),
                    "mean_kurtosis_after": float(per_df["kurtosis_after"].mean()),
                    "median_kurtosis_after": float(per_df["kurtosis_after"].median()),
                    "mean_skew_after": float(per_df["skew_after"].mean()),
                    "mean_pct_outliers": float(per_df["pct_outliers_stage1"].mean())
                }
                results.append(agg)

    res_df = pd.DataFrame(results)
    csv_path = f"{out_prefix}_summary.csv"
    res_df.to_csv(csv_path, index=False)

    # Simple diagnostic plot: mean_rmse_after vs mean_kurtosis_after colored by deg (save)
    plt.figure(figsize=(8, 6))
    for deg in sorted(res_df["deg"].unique()):
        sub = res_df[res_df["deg"] == deg]
        plt.scatter(sub["mean_kurtosis_after"],
                    sub["mean_rmse_after"], label=f"deg={deg}", s=60)
        for _, row in sub.iterrows():
            plt.annotate(f"t{row['tukey_c']}-h{row['huber_c']}",
                         (row["mean_kurtosis_after"], row["mean_rmse_after"]), fontsize=8)
    plt.xlabel("Mean Kurtosis After")
    plt.ylabel("Mean RMSE After")
    plt.title("Grid search: RMSE vs Kurtosis (per parameter combo)")
    plt.legend()
    png1 = f"{out_prefix}_rmse_vs_kurtosis.png"
    plt.savefig(png1, dpi=200)
    plt.close()

    # Heatmap style: pivot by tukey_c x huber_c for a chosen deg (first deg)
    choose_deg = sorted(res_df["deg"].unique())[0]
    pivot = res_df[res_df["deg"] == choose_deg].pivot(
        index="tukey_c", columns="huber_c", values="mean_kurtosis_after")
    plt.figure(figsize=(6, 5))
    plt.imshow(pivot.values, aspect='auto', origin='lower')
    plt.colorbar(label='mean_kurtosis_after')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"Mean Kurtosis After (deg={choose_deg})")
    png2 = f"{out_prefix}_kurtosis_heatmap_deg{choose_deg}.png"
    plt.savefig(png2, dpi=200)
    plt.close()

    return res_df
