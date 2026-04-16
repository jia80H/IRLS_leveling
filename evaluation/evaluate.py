from typing import Dict, Any
from scipy.stats import kurtosis, skew
import pandas as pd
import sqlite3
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ---------- 频谱条带能量比 ----------
from typing import Literal


def spectral_stripe_ratio(npz_path, angle_deg=0.0, bandwidth=0.05):
    """
    计算条带能量占比：stripe_power / total_power
    angle_deg: 条带走向（度），0 表示 x 方向（东西），90 表示 y 方向（南北）
    bandwidth: 控制在频域上投影宽度（与 remove_stripes_fft 一致）
    返回 (stripe_power, total_power, ratio)
    """
    data = np.load(npz_path)
    z = data["z"]
    fft_data = np.fft.fftshift(np.fft.fft2(z))
    total_power = np.sum(np.abs(fft_data)**2)

    nx, ny = z.shape
    u = np.linspace(-0.5, 0.5, ny)
    v = np.linspace(-0.5, 0.5, nx)
    uu, vv = np.meshgrid(u, v)
    theta = np.deg2rad(angle_deg)
    perp = -uu * np.sin(theta) + vv * np.cos(theta)

    # 定义条带频带掩模（窄带）
    stripe_mask = np.exp(- (perp**2) / (2 * bandwidth**2))
    stripe_power = np.sum((np.abs(fft_data)**2) * stripe_mask)

    ratio = stripe_power / total_power if total_power != 0 else np.nan
    return float(stripe_power), float(total_power), float(ratio)


# ---------- 方差各向异性比 ----------
def anisotropy_ratio(npz_path):
    """
    计算沿向与跨向差分方差比。
    假设 z 的第 0 维为 'y'（行），第 1 维为 'x'（列）。
    返回 var_along / var_across
    """
    data = np.load(npz_path)
    z = data["z"]
    # 差分：沿航向（按行方向差分）和跨航向（按列差分）
    diff_along = np.diff(z, axis=0)   # 沿行（y方向）
    diff_across = np.diff(z, axis=1)  # 沿列（x方向）
    var_along = np.nanvar(diff_along)
    var_across = np.nanvar(diff_across)
    ratio = var_along / var_across if var_across != 0 else np.nan
    return float(var_along), float(var_across), float(ratio)


# ---------- 基于格网生成“虚拟交点 / 邻线 RMSE” ----------
def virtual_line_rmse(survey, corrected_grid_npz, n_sample_per_line=200):
    """
    用法：
      survey: MagSurveyData 实例（含 survey.survey_lines dict）
      corrected_grid_npz: remove_stripes_fft 输出的去条带格网 npz
    思路：
      - 对每条测线，沿其点的投影位置在格网上插值 n_sample_per_line 个点，得到“虚拟测线”序列
      - 对相邻测线计算在相同投影参数处的差值，统计 RMS（类似交点差）
    返回：
      dict: {'pairs': [(lineA,lineB,rmse), ...], 'global_rmse': ...}
    """
    data = np.load(corrected_grid_npz)
    Xg = data["x"]
    Yg = data["y"]
    Zg = data["z"]
    # RegularGridInterpolator expects increasing axes
    x_coords = Xg[0, :]
    y_coords = Yg[:, 0]
    interp = RegularGridInterpolator(
        (y_coords, x_coords), Zg, bounds_error=False, fill_value=np.nan)

    # get sorted line ids by center Y (or X)
    lines = survey.survey_lines  # dict {id: df}
    # compute line centers and sort by perpendicular coordinate
    centers = []
    for lid, df in lines.items():
        cx = df[survey.x_col].mean()
        cy = df[survey.y_col].mean()
        centers.append((lid, cx, cy))
    # sort by y (assume predominant flight in y) - this is heuristic
    centers_sorted = sorted(centers, key=lambda t: (t[2], t[1]))
    pairs = []
    all_diffs = []

    # For each adjacent pair, sample along the longer of the two extents
    for i in range(len(centers_sorted)-1):
        lid1 = centers_sorted[i][0]
        lid2 = centers_sorted[i+1][0]
        df1 = lines[lid1].sort_values(by=[survey.y_col, survey.x_col])
        df2 = lines[lid2].sort_values(by=[survey.y_col, survey.x_col])

        # parameterize along-track by normalized distance s in [0,1]
        def sample_line(df, n):
            pts = df[[survey.x_col, survey.y_col]].values
            # cumulative distance
            d = np.cumsum(np.r_[0, np.linalg.norm(
                np.diff(pts, axis=0), axis=1)])
            if d[-1] == 0:
                return np.array([]), np.array([])
            s_vals = np.linspace(0, d[-1], n)
            xs = np.interp(s_vals, d, pts[:, 0])
            ys = np.interp(s_vals, d, pts[:, 1])
            return xs, ys

        xs1, ys1 = sample_line(df1, n_sample_per_line)
        xs2, ys2 = sample_line(df2, n_sample_per_line)
        if xs1.size == 0 or xs2.size == 0:
            continue

        # Interpolate grid at those sample points
        pts1 = np.vstack([ys1, xs1]).T
        pts2 = np.vstack([ys2, xs2]).T
        z1 = interp(pts1)
        z2 = interp(pts2)
        # Keep only positions where both valid
        mask = ~np.isnan(z1) & ~np.isnan(z2)
        if np.count_nonzero(mask) < 10:
            continue
        diff = z1[mask] - z2[mask]
        rmse = np.sqrt(np.mean(diff**2))
        pairs.append((lid1, lid2, float(rmse)))
        all_diffs.append(diff)

    # global RMSE across all pairs
    if len(all_diffs) == 0:
        global_rmse = np.nan
    else:
        all_vals = np.hstack(all_diffs)
        global_rmse = float(np.sqrt(np.mean(all_vals**2)))

    return {'pairs': pairs, 'global_rmse': global_rmse}


def apply_grid_to_points(db_path, table_name, x_col, y_col, mag_col, corrected_npz, out_table="mag_corrected"):
    data = np.load(corrected_npz)
    Xg = data["x"]
    Yg = data["y"]
    Zg = data["z"]
    interp = RegularGridInterpolator(
        (Yg[:, 0], Xg[0, :]), Zg, bounds_error=False, fill_value=np.nan)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"SELECT rowid, {x_col}, {y_col}, {mag_col} FROM {table_name}", conn)
    pts = np.vstack([df[y_col].values, df[x_col].values]).T
    z_corr = interp(pts)
    df["Mag_corrected"] = z_corr
    # 写回数据库：可以新建表或更新
    df.to_sql(out_table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"已将 {len(df)} 个点的格网改正值写入表 {out_table}")


def evaluate_leveling(
    db_path: str,
    intersection_table: str = 'Tie_intersection',
    grad_thresh: float = 600,
    mask_channel: str = 'MASK'
):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {intersection_table}", conn)
    conn.close()

    # 条件筛选
    valid = df['CROSS_DIFF'].notna()

    if grad_thresh is not None:
        valid &= (df['TIE_GRAD'].abs() < grad_thresh) & (
            df['LINE_GRAD'].abs() < grad_thresh)

    if mask_channel in df.columns:
        valid &= ~df[mask_channel].isna()

    df_valid = df[valid]

    if df_valid.empty:
        return "无有效交点，无法评估调平效果。"

    diff = df_valid['CROSS_DIFF']

    return {
        '有效交点数': len(diff),
        'RMSE': np.sqrt(np.mean(diff ** 2)),
        'MAE': np.mean(np.abs(diff)),
        '最大误差': np.max(np.abs(diff)),
        '均值': np.mean(diff),
        '标准差': np.std(diff)
    }


def evaluate_intersection_metrics(
    db_path: str,
    table_name: str,
    diff_channel: str = 'CROSS_DIFF',
    # 使用'Fisher'或'Pearson'定义. True=Normal is 0; False=Normal is 3
    kurtosis_fisher: bool = False
) -> Dict[str, Any]:
    """
    计算并评估交点表（如Tie_intersection）中特定错合差通道的统计指标。

    这些指标用于量化磁力数据在不同调平阶段（调平前、调平后）的精度和统计特性。

    :param db_path: SQLite数据库路径。
    :param table_name: 存储交点信息的数据库表名 (e.g., 'Tie_intersection')。
    :param diff_channel: 包含错合差数值的通道名称 (e.g., 'CROSS_DIFF')。
    :param kurtosis_fisher: 峰度计算模式。False (默认) 表示正态分布峰度为 3 (Pearson定义)，
                            与您文章中“峰度趋近于3”的验证标准一致。
    :return: 包含统计指标（STD, Max, Kurtosis等）的字典。
    """
    conn = sqlite3.connect(db_path)

    try:
        # 1. 读取数据并清理
        df_inter = pd.read_sql(
            f"SELECT {diff_channel} FROM {table_name}", conn)
        diff_data = df_inter[diff_channel].dropna().values

        if diff_data.size == 0:
            print(f"⚠️ 表 {table_name} 或通道 {diff_channel} 中没有有效数据。")
            return {}

        # 2. 计算核心统计指标
        mean_diff = np.mean(diff_data)
        std_diff = np.std(diff_data)
        max_abs_diff = np.max(np.abs(diff_data))
        min_diff = np.min(diff_data)
        max_diff = np.max(diff_data)

        # 3. 计算稳健统计指标 (峰度和偏度)
        # 您文章中要求“峰度值趋近于 3”，因此我们使用 Pearson 定义 (kurtosis_fisher=False)
        kurt_diff = kurtosis(diff_data, fisher=kurtosis_fisher)
        skew_diff = skew(diff_data)

        # 4. 汇总结果
        results = {
            'Intersection_Count': len(diff_data),
            'Mean_Difference (nT)': mean_diff,
            'STD_Difference (nT)': std_diff,
            'Max_Absolute_Difference (nT)': max_abs_diff,
            'Min_Difference (nT)': min_diff,
            'Max_Difference (nT)': max_diff,
            'Kurtosis (Peakness)': kurt_diff,
            'Skewness (Asymmetry)': skew_diff
        }

        # 5. 打印报告
        print(f"\n--- 交点错合差统计报告 ({table_name} / {diff_channel}) ---")
        print(f"总交点数: {results['Intersection_Count']}")
        print(f"平均差值 (Mean): {results['Mean_Difference (nT)']:.4f} nT")
        print(f"标准差 (STD): {results['STD_Difference (nT)']:.4f} nT")
        print(
            f"最大绝对差值 (Max Abs): {results['Max_Absolute_Difference (nT)']:.4f} nT")
        print(
            f"峰度 (Kurtosis): {results['Kurtosis (Peakness)']:.4f} (目标值: 3.0)")
        print(
            f"偏度 (Skewness): {results['Skewness (Asymmetry)']:.4f} (目标值: 0.0)")
        print("----------------------------------------------------------\n")

        return results

    except Exception as e:
        print(f"处理交点表 {table_name} 时发生错误: {e}")
        return {}
    finally:
        conn.close()


def evaluate_intersection_metrics_refined(
    db_path: str,
    table_name: str,
    diff_channel: str = 'CROSS_DIFF',
    kurtosis_fisher: bool = False,
    reject_outliers: bool = True,  # 新增：是否剔除异常值
    method: Literal['3sigma', 'MAD'] = 'MAD'  # 新增：剔除方法
) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        df_inter = pd.read_sql(
            f"SELECT {diff_channel} FROM {table_name}", conn)
        raw_data = df_inter[diff_channel].dropna().values

        if raw_data.size == 0:
            return {}

        # --- 新增：异常值剔除逻辑 ---
        if reject_outliers:
            if method == '3sigma':
                mean_val = np.mean(raw_data)
                std_val = np.std(raw_data)
                mask = (raw_data > mean_val - 3 *
                        std_val) & (raw_data < mean_val + 3 * std_val)
            else:  # MAD (Median Absolute Deviation)
                median = np.median(raw_data)
                mad = np.median(np.abs(raw_data - median))
                # 1.4826 是正态分布的缩放因子
                threshold = 3 * (1.4826 * mad)
                mask = (raw_data > median -
                        threshold) & (raw_data < median + threshold)

            diff_data = raw_data[mask]
            outlier_count = len(raw_data) - len(diff_data)
        else:
            diff_data = raw_data
            outlier_count = 0
        # --------------------------

        # 计算核心统计指标
        results = {
            'Intersection_Count': len(diff_data),
            'Outliers_Removed': outlier_count,
            'Mean_Difference (nT)': np.mean(diff_data),
            'STD_Difference (nT)': np.std(diff_data),
            'Max_Absolute_Difference (nT)': np.max(np.abs(diff_data)),
            'Kurtosis (Peakness)': kurtosis(diff_data, fisher=kurtosis_fisher),
            'Skewness (Asymmetry)': skew(diff_data)
        }

        # 打印报告时增加异常值剔除信息
        print(f"剔除异常值数量: {outlier_count} (方法: {method})")
        print(f"标准差 (STD): {results['STD_Difference (nT)']:.4f} nT")
        # print(
        #     f"峰度 (Kurtosis): {results['Kurtosis (Peakness)']:.4f} (目标值: 3.0)")

        return results
    except Exception as e:
        print(f"错误: {e}")
        return {}
    finally:
        conn.close()
