"""
联络测线调平
1.进行Tie_lines intersection其中data_ch为没有调平的原始数据,Tie_lines=True, All_lines=False, output_tab = 'Tie_intersection'.
2.进行Tie_lines load_correction其中Tie_lines=True, All_lines=False, intersection_tab = 'Tie_intersection'.
3.进行Tie_lines statistically_leveling 其中Tie_lines=True, All_lines=False, selected_lines=None.
常规测线调平
1.进行all_lines intersection其中data_ch为first_leveled_data_ch,Tie_lines=False, All_lines=True, output_tab = 'All_intersection'.
2.进行all_lines load_correction其中Tie_lines=False, All_lines=True, intersection_tab = 'All_intersection'.
3.进行all_lines statistically_leveling 其中Tie_lines=False, All_lines=True, selected_lines=None, intersection_tab = 'All_intersection'.


"""
from scipy.stats import kurtosis, skew, median_abs_deviation
from typing import Literal, Dict, Any
from numpy.polynomial.polynomial import Polynomial
from shapely.geometry import LineString
from typing import Literal
import numpy as np
import pandas as pd
import sqlite3
from scipy.spatial import cKDTree


from scipy.stats import kurtosis


def calculate_metrics_for_sci(db_path, result_tab, original_mask_tab, diff_col='CROSS_DIFF', baseline_sigma=None):
    """
    输入处理后的结果表和原始 Mask 表，输出 SCI 级别的评价指标。

    参数:
    db_path: 数据库路径
    result_tab: 调平后的交点表（Scenario 1-4 的输出）
    original_mask_tab: 原始生成的带 Mask 的表（用于统一剔除点）
    diff_col: 调平后的残差列名
    baseline_sigma: Scenario 1 的 sigma，用于计算改善率
    """
    conn = sqlite3.connect(db_path)

    # 读取调平后的数据和原始的 Mask 掩码
    df_res = pd.read_sql(f"SELECT * FROM {result_tab}", conn)
    df_mask = pd.read_sql(f"SELECT MASK FROM {original_mask_tab}", conn)

    # 合并 Mask 以确保剔除的是同样的点
    df_res['ORIGINAL_MASK'] = df_mask['MASK']

    # 统一剔除点：只保留原始 Mask 为 1.0 的数据
    valid_data = df_res[df_res['ORIGINAL_MASK']
                        == 1.0][diff_col].dropna().values

    if len(valid_data) == 0:
        conn.close()
        return "Error: No valid data after masking."

    # 1. Standard Deviation (sigma)
    std_val = np.std(valid_data, ddof=1)

    # 2. Mean Absolute Error (MAE)
    mae_val = np.mean(np.abs(valid_data - np.mean(valid_data)))

    # 3. Kurtosis (K)
    # 注意：计算峰度时建议使用该场景下的全量数据（包含异常点），以体现算法对分布的修正能力
    # full_data = df_res[diff_col].dropna().values
    kurt_val = kurtosis(valid_data, fisher=False)  # Gaussian = 3

    metrics = {
        "Scenario": result_tab,
        "Std. Dev (sigma)": round(std_val, 4),
        "MAE": round(mae_val, 4),
        "Kurtosis (K)": round(kurt_val, 4),
        "N (Points)": len(valid_data)
    }

    # 4. Improvement Ratio (eta)
    if baseline_sigma:
        eta = ((baseline_sigma - std_val) / baseline_sigma) * 100
        metrics["Improvement (%)"] = f"{eta:.2f}%"

    conn.close()
    return metrics


def tieline_intersection_tree(db_path, LineID, xch, ych, data_ch, output_tab='Tie_intersection',
                              main_tabname='mag_data', intersection_tolerance=0):
    """
    计算 TieLine 与普通测线的交点信息，并保存到数据库中。
    字段完全对齐 Oasis Montaj 官方字段说明。
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {main_tabname}", conn)

    # 标准化线号为字符串
    df[LineID] = df[LineID].astype(str)
    df_tie = df[df[LineID].str.startswith('T')].copy()
    df_line = df[df[LineID].str.startswith('L')].copy()

    results = []

    # 构建测线空间索引
    for line_id, line_df in df_line.groupby(LineID):
        coords = line_df[[xch, ych]].values
        tree = cKDTree(coords)

        line_vals = line_df[data_ch].values
        line_fids = line_df.index.to_numpy()
        line_grads = np.gradient(line_vals)

        # 遍历每条切割线
        for tie_id, tie_df in df_tie.groupby(LineID):
            tie_coords = tie_df[[xch, ych]].values
            tie_vals = tie_df[data_ch].values
            tie_fids = tie_df.index.to_numpy()
            tie_grads = np.gradient(tie_vals)

            dists, idxs = tree.query(
                tie_coords, distance_upper_bound=intersection_tolerance)

            for i, (dist, idx) in enumerate(zip(dists, idxs)):
                if idx >= len(coords) or np.isinf(dist):
                    continue

                tx, ty = tie_coords[i]
                lx, ly = coords[idx]

                results.append({
                    'X': tx,
                    'Y': ty,
                    'TIE': tie_id,
                    'TIE_FID': tie_fids[i],
                    'TIE_LEVEL': tie_vals[i],
                    'TIE_GRAD': tie_grads[i],
                    'LINE': line_id,
                    'LINE_FID': line_fids[idx],
                    'LINE_LEVEL': line_vals[idx],
                    'LINE_GRAD': line_grads[idx],
                    'Intersection_dX': lx - tx,
                    'Intersection_dY': ly - ty,
                    'Intersection_dDist': np.sqrt((lx - tx) ** 2 + (ly - ty) ** 2),
                    'Intersection_dData': tie_vals[i] - line_vals[idx],
                    'CROSS_DIFF': line_vals[idx] - tie_vals[i],
                })

    df_out = pd.DataFrame(results)
    df_out.to_sql(output_tab, conn, if_exists='replace')
    conn.close()
    print(f"✅ 共识别交点 {len(df_out)} 个，已保存至表 {output_tab}")


def tieline_intersection_shapely(db_path, LineID, xch, ych, data_ch,
                                 output_tab='Tie_intersection',
                                 main_tabname='mag_data',
                                 intersection_tolerance=0):
    # 连接数据库
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {main_tabname}", conn)

    # 分开 Tie 与 Line
    df[LineID] = df[LineID].astype(str)
    tie_lines = df[df[LineID].str.startswith('T')].copy()
    reg_lines = df[df[LineID].str.startswith('L')].copy()

    results = []

    # 遍历每条 Tie 线
    for tie_id, tie_df in tie_lines.groupby(LineID):
        tie_coords = tie_df[[xch, ych]].values
        tie_line = LineString(tie_coords)
        tie_vals = tie_df[data_ch].values
        tie_fids = tie_df.index.values
        tie_grads = np.gradient(tie_vals)

        for reg_id, reg_df in reg_lines.groupby(LineID):
            reg_coords = reg_df[[xch, ych]].values
            reg_line = LineString(reg_coords)
            reg_vals = reg_df[data_ch].values
            reg_fids = reg_df.index.values
            reg_grads = np.gradient(reg_vals)

            inter = tie_line.intersection(reg_line)

            if inter.is_empty:
                continue

            points = []
            if inter.geom_type == 'Point':
                points = [inter]
            elif inter.geom_type == 'MultiPoint':
                points = list(inter.geoms)

            for pt in points:
                x, y = pt.x, pt.y

                # 最近 tie 点
                tie_dists = np.linalg.norm(tie_coords - [x, y], axis=1)
                tie_idx = np.argmin(tie_dists)
                if intersection_tolerance > 0 and tie_dists[tie_idx] > intersection_tolerance:
                    continue

                # 最近 reg 点
                reg_dists = np.linalg.norm(reg_coords - [x, y], axis=1)
                reg_idx = np.argmin(reg_dists)
                if intersection_tolerance > 0 and reg_dists[reg_idx] > intersection_tolerance:
                    continue

                # 提取各项数据
                tie_fid = tie_fids[tie_idx]
                tie_level = tie_vals[tie_idx]
                tie_grad = tie_grads[tie_idx]

                reg_fid = reg_fids[reg_idx]
                reg_level = reg_vals[reg_idx]
                reg_grad = reg_grads[reg_idx]

                dx = reg_coords[reg_idx][0] - tie_coords[tie_idx][0]
                dy = reg_coords[reg_idx][1] - tie_coords[tie_idx][1]
                dist = np.sqrt(dx**2 + dy**2)

                results.append({
                    'X': x,
                    'Y': y,
                    'TIE': tie_id,
                    'TIE_FID': tie_fid,
                    'TIE_LEVEL': tie_level,
                    'TIE_GRAD': tie_grad,
                    'LINE': reg_id,
                    'LINE_FID': reg_fid,
                    'LINE_LEVEL': reg_level,
                    'LINE_GRAD': reg_grad,
                    'Intersection_dX': dx,
                    'Intersection_dY': dy,
                    'Intersection_dDist': dist,
                    'Intersection_dData': tie_level - reg_level,
                    'CROSS_DIFF': reg_level - tie_level
                })

    out_df = pd.DataFrame(results)
    out_df.to_sql(output_tab, conn, if_exists='replace', index=False)
    conn.close()

    print(f"交点计算完成，结果保存至表 {output_tab}，共 {len(out_df)} 个交点。")


def tieline_intersection(db_path, LineID, xch, ych, data_ch, output_tab='Tie_intersection',
                         main_tabname='mag_data', intersection_tolerance=10):
    """
    使用 Shapely 计算几何交点，用 KDTree 匹配最近采样点，结果字段严格对齐 Oasis Montaj 标准。
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {main_tabname}", conn)

    df[LineID] = df[LineID].astype(str)
    df_tie = df[df[LineID].str.startswith('T')].copy()
    df_line = df[df[LineID].str.startswith('L')].copy()

    results = []

    for tie_id, tie_df in df_tie.groupby(LineID):
        tie_line = LineString(tie_df[[xch, ych]].values)
        tie_vals = tie_df[data_ch].values
        tie_coords = tie_df[[xch, ych]].values
        tie_fids = tie_df.index.to_numpy()
        tie_grads = np.gradient(tie_vals)

        for line_id, line_df in df_line.groupby(LineID):
            line_line = LineString(line_df[[xch, ych]].values)
            inter = tie_line.intersection(line_line)
            if inter.is_empty:
                continue

            if inter.geom_type == 'Point':
                inters = [inter]
            elif inter.geom_type == 'MultiPoint':
                inters = list(inter.geoms)
            else:
                continue

            line_coords = line_df[[xch, ych]].values
            line_vals = line_df[data_ch].values
            line_fids = line_df.index.to_numpy()
            line_grads = np.gradient(line_vals)
            line_tree = cKDTree(line_coords)

            for pt in inters:
                x, y = pt.x, pt.y
                tie_dist = np.linalg.norm(
                    tie_coords - np.array([x, y]), axis=1)
                tie_idx = np.argmin(tie_dist)

                dist, line_idx = line_tree.query(
                    [x, y], distance_upper_bound=intersection_tolerance)
                if line_idx >= len(line_coords) or np.isinf(dist):
                    continue

                lx, ly = line_coords[line_idx]
                results.append({
                    'X': x,
                    'Y': y,
                    'TIE': tie_id,
                    'TIE_FID': tie_fids[tie_idx],
                    'TIE_LEVEL': tie_vals[tie_idx],
                    'TIE_GRAD': tie_grads[tie_idx],
                    'LINE': line_id,
                    'LINE_FID': line_fids[line_idx],
                    'LINE_LEVEL': line_vals[line_idx],
                    'LINE_GRAD': line_grads[line_idx],
                    'Intersection_dX': lx - x,
                    'Intersection_dY': ly - y,
                    'Intersection_dDist': np.sqrt((lx - x)**2 + (ly - y)**2),
                    'Intersection_dData': tie_vals[tie_idx] - line_vals[line_idx],
                    'CROSS_DIFF': line_vals[line_idx] - tie_vals[tie_idx]
                })

    df_out = pd.DataFrame(results)
    df_out.to_sql(output_tab, conn, if_exists='replace')
    conn.close()
    print(f"交点计算完成，结果保存至表 {output_tab}，共 {len(df_out)} 个交点。")


def load_correction(db_path,
                    intersection_table='Tie_intersection',
                    main_table='mag_data',
                    line_id_col='LineID',
                    max_grad=None,
                    mask_channel='MASK',
                    output_level_ch='CROSS_LEVEL',
                    output_diff_ch='CROSS_DIFF',
                    output_grad_ch='CROSS_GRAD',
                    process_line_types: Literal['TIE', 'LINE'] = 'TIE'
                    ):
    """
    将交点校正信息从Tie_intersection表写入mag_data表中的各个测线。
    支持最大梯度阈值和mask筛选，字段对齐Oasis Montaj的XLEVEL GX参数说明。
    """
    conn = sqlite3.connect(db_path)
    df_inter = pd.read_sql(f"SELECT * FROM {intersection_table}", conn)
    df_main = pd.read_sql(f"SELECT rowid, * FROM {main_table}", conn)

    # 应用梯度筛选
    if max_grad is not None:
        df_inter = df_inter[
            (df_inter['TIE_GRAD'].abs() <= max_grad) &
            (df_inter['LINE_GRAD'].abs() <= max_grad)
        ]

    # 应用 MASK 筛选（非 dummy）
    # 如果存在 mask_channel 这一列，就把这一列中为空（NaN）的行全部删掉，只保留有效数据。
    if mask_channel in df_inter.columns:
        df_inter = df_inter[~df_inter[mask_channel].isna()]

    # 初始化新通道列
    for col in [output_level_ch, output_diff_ch, output_grad_ch]:
        if col and col not in df_main.columns:
            df_main[col] = np.nan

    # 如果是重新处理 TIE，则清空已有结果
    if process_line_types == 'TIE':
        for col in [output_level_ch, output_diff_ch, output_grad_ch]:
            if col in df_main.columns:
                df_main[col] = np.nan

    # 遍历交点
    update_count = 0
    for _, row in df_inter.iterrows():
        line_id = row[f'{process_line_types}']
        fid = row[f'{process_line_types}_FID']

        match = (df_main[line_id_col] == line_id) & (df_main.index == fid)
        if match.sum() != 1:
            continue  # 跳过找不到匹配的观测点

        idx = df_main[match].index[0]
        if output_level_ch:
            df_main.at[idx, output_level_ch] = row['TIE_LEVEL']
        if output_diff_ch:
            df_main.at[idx, output_diff_ch] = row['CROSS_DIFF']
        if output_grad_ch:
            df_main.at[idx, output_grad_ch] = row['TIE_GRAD']
        update_count += 1

    print(f"共更新 {update_count} 个交点至 {main_table} 表")

    # 保存结果
    df_main.drop(columns=['rowid'], errors='ignore').to_sql(
        main_table, conn, if_exists='replace', index=False)
    conn.close()


def load_correction_tie(db_path,
                        intersection_table='Tie_intersection',
                        main_table='mag_data',
                        line_id_col='LineID',
                        max_grad=None,
                        mask_channel='MASK',
                        output_level_ch='TIE_LEVEL',
                        output_diff_ch='CROSS_DIFF',
                        output_grad_ch='TIE_GRAD'
                        ):
    """
    将交点校正信息从Tie_intersection表写入mag_data表中的各个测线。
    支持最大梯度阈值和mask筛选，字段对齐Oasis Montaj的XLEVEL GX参数说明。
    """
    conn = sqlite3.connect(db_path)
    df_inter = pd.read_sql(f"SELECT * FROM {intersection_table}", conn)
    df_main = pd.read_sql(f"SELECT rowid, * FROM {main_table}", conn)

    # 应用梯度筛选
    if max_grad is not None:
        df_inter = df_inter[
            (df_inter['TIE_GRAD'].abs() <= max_grad) &
            (df_inter['LINE_GRAD'].abs() <= max_grad)
        ]

    # 应用 MASK 筛选（非 dummy）
    if mask_channel in df_inter.columns:
        df_inter = df_inter[~df_inter[mask_channel].isna()]

    # 初始化新通道列
    for col in [output_level_ch, output_diff_ch, output_grad_ch]:
        if col and col not in df_main.columns:
            df_main[col] = np.nan

    # 遍历交点
    update_count = 0
    for _, row in df_inter.iterrows():
        line_id = row['TIE']
        fid = row['TIE_FID']

        match = (df_main[line_id_col] == line_id) & (df_main.index == fid)
        if match.sum() != 1:
            continue  # 跳过找不到匹配的观测点

        idx = df_main[match].index[0]
        if output_level_ch:
            df_main.at[idx, output_level_ch] = row['TIE_LEVEL']
        if output_diff_ch:
            df_main.at[idx, output_diff_ch] = row['CROSS_DIFF']
        if output_grad_ch:
            df_main.at[idx, output_grad_ch] = row['TIE_GRAD']
        update_count += 1

    print(f"共更新 {update_count} 个交点至 {main_table} 表")

    # 保存结果
    df_main.drop(columns=['rowid'], errors='ignore').to_sql(
        main_table, conn, if_exists='replace', index=False)
    conn.close()


def intersection(db_path, LineID, xch, ych, data_ch, output_tab, main_tabname='mag_data', intersection_tolerance=0, Tie_lines=True, All_lines=False):
    """
    计算测线交点, 并保存到数据库, 当Tie_lines = True,All_lines = False是计算出所有切割线与测线的的交点。生成的表名为Tie。结果样表表头如下：
    | LineID | X | Y | Data | Mask | Tie | TIE_FID | TIE_LEVEL | tIE_GRAD | lINE_ID_CROSS | LINE_FID | LINE_LEVEL | LINE_GRAD | CROSS_DIFF |
    :param db_path: 数据库路径
    :param LineID: 测线ID列名
    :param xch: X坐标列名
    :param ych: Y坐标列名
    :param data_ch: 数据列名
    :param output_tab: 输出表名
    :param intersection_tolerance: 交点容差
    :param Tie_lines: 是否计算Tie线
    :param All_lines: 是否计算所有测线的交点
    """
    pass


# def load_correction(db_path, LineID, xch, ych, unleveled_data_ch, intersection_tab,
#                     main_tabname='mag_data',
#                     Maximum_gradient=0,
#                     Tie_lines=True, All_lines=False, selected_lines=None,
#                     output_cross_leve_channel='cross_level', output_diff_channel='cross_diff', output_cross_grad_channel='cross_grad'):
#     """
#     导入测线交点数据，并计算交点的修正值。结果保存到数据库中, 没有数据的暂时填nan。
#     :param db_path: 数据库路径
#     :param LineID: 测线ID列名
#     :param xch: X坐标列名
#     :param ych: Y坐标列名
#     :param unleveled_data_ch: 未修正数据列名
#     :param main_tabname: 表名
#     :param intersection_tab: 交点表名
#     :param Maximum_gradient: Z/fid 最大梯度
#     :param Tie_lines: 是否计算Tie线
#     :param All_lines: 是否计算所有测线的交点
#     :param selected_lines: 选择的测线ID
#     :param output_cross_leve_channel: 交点cross_leve_channel列名
#     :param output_diff_channel: 交点diff_channel列名
#     :param output_cross_grad_channel: 交点grad_channel列名
#     """
#     pass


def robust_polynomial_fit(x: np.ndarray, y: np.ndarray, deg: int,
                          max_iter: int = 20, tolerance: float = 1e-6,
                          huber_c: float = 1.345, debug: bool = False) -> Polynomial:
    """
    使用基于 Huber 损失的迭代重加权最小二乘法 (IRLS) 进行稳健多项式拟合。

    :param debug: 如果为 True，将在每次迭代时打印调试信息。
    """
    if len(x) < deg + 1:
        if debug:
            print("Debug: 样本数不足，返回均值。")
        return Polynomial([np.mean(y)])

    # 构建范德蒙矩阵 [x^deg, ..., x^1, x^0]
    X = np.vander(x, deg + 1)
    weights = np.ones_like(y)
    old_coef = None

    # 跟踪收敛状态
    is_converged = False

    for iteration in range(1, max_iter + 1):
        # 1. 加权最小二乘拟合
        W = np.sqrt(weights)

        try:
            # 求解 (W * X) * coef = (W * y)
            coef, _, _, _ = np.linalg.lstsq(
                X * W[:, np.newaxis], y * W, rcond=None)
        except np.linalg.LinAlgError:
            if debug:
                print(f"Debug: 迭代 {iteration}: 线性代数错误 (奇异矩阵)。")
            break  # 拟合失败

        # 检查收敛
        if old_coef is not None and np.linalg.norm(coef - old_coef) < tolerance:
            is_converged = True
            break

        # 2. 计算残差和稳健尺度
        residuals = y - np.dot(X, coef)

        # 使用 MAD 稳健地估计尺度（Median Absolute Deviation）
        s = 1.4826*median_abs_deviation(residuals)

        if s == 0:
            if debug:
                print(f"Debug: 迭代 {iteration}: 稳健尺度 s=0，拟合完美或数据为常数。")
            is_converged = True
            break

        # 3. Huber 损失函数权重更新
        u = np.abs(residuals / s)  # 标准化残差
        # Huber 权重：|u| <= c: w=1 (L2惩罚); |u| > c: w=c/|u| (L1惩罚)
        weights = np.where(u <= huber_c, 1.0, huber_c / u)

        # 4. 调试输出
        if debug:
            print(f"Iter {iteration:02d}: STD(Res)={np.std(residuals):.2f}, Scale(s)={s:.2f}, "
                  f"Weights: Min={np.min(weights):.4f}, Max={np.max(weights):.4f}")

        old_coef = coef

    if debug and not is_converged:
        print(f"Debug: IRLS 未在 {max_iter} 次迭代内收敛。")

    # coef 是 [c_deg, ..., c_0], Polynomial([c_0, ..., c_deg])
    return Polynomial(coef[::-1])


def statistical_level(
    db_path: str,
    table: str,
    line_id_col: str = 'LineID',
    type_filter: str = 'LINE',  # TIE, LINE 或 ALL
    input_ch: str = 'Data',
    diff_ch: str = 'CROSS_DIFF',
    output_ch: str = 'LEVELLED',
    trend_order: int = 1,
    robust_mode: bool = False,
    huber_c: float = 1.5,
    max_iter: int = 40,  # <--- 新增最大迭代次数
    debug: bool = False,  # <--- 新增调试开关
    trend_out_ch: str = 'TREND',
):
    """
    进行测线统计调平（OLS 或 IRLS 稳健拟合）。
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT rowid, * FROM {table}", conn)

    # ... (筛选逻辑) ...
    df[line_id_col] = df[line_id_col].astype(str)
    all_lines = df[line_id_col].unique()
    if type_filter == 'TIE':
        selected_lines = [l for l in all_lines if l.startswith('T')]
    elif type_filter == 'LINE':
        selected_lines = [l for l in all_lines if l.startswith('L')]
    else:
        selected_lines = all_lines

    # ... (初始化输出通道) ...
    if output_ch not in df.columns:
        df[output_ch] = df[input_ch]
    if trend_out_ch and trend_out_ch not in df.columns:
        df[trend_out_ch] = np.nan

    # 3. 对每条测线拟合校正趋势
    df_diff = df.loc[df[diff_ch].notna(), [line_id_col, diff_ch]].copy()
    corrections = {}

    # 拟合方法选择
    fit_type = "OLS"
    if robust_mode:
        def fit_function(x, y, deg): return robust_polynomial_fit(
            x, y, deg=deg, huber_c=huber_c, max_iter=max_iter, debug=debug
        )
        fit_type = f"IRLS-Huber(c={huber_c}, MaxIter={max_iter})"
    else:
        def fit_function(x, y, deg): return Polynomial.fit(
            x, y, deg=deg).convert()

    print(f"ℹ️ 正在使用 {fit_type} 对 {type_filter} 测线进行调平...")

    for line in selected_lines:
        line_diffs = df_diff[df_diff[line_id_col] == line]
        if line_diffs.empty:
            continue

        x = df.loc[line_diffs.index]['rowid'].values.astype(float)
        y = line_diffs[diff_ch].values.astype(float)

        if debug:
            print(f"\n--- DEBUG START: Line {line} ---")

        try:
            p = fit_function(x, y, deg=trend_order)
            corrections[line] = p
        except Exception as e:
            print(f"Warning: Line {line} {fit_type} fit failed. Error: {e}")
            continue

        if debug:
            print(f"--- DEBUG END: Line {line} ---")
    # 添加的抑制短波长
    res = y - p(x)

    # 二阶差分平滑（只压短波）
    lambda_hf = 0.2  # 高频惩罚强度，可调
    d2 = np.diff(res, n=2)
    d2_smooth = np.convolve(d2, np.ones(3)/3, mode='same')

    # 重建平滑后的残差
    res_smooth = np.concatenate(
        [[res[0], res[1]], res[2:] - lambda_hf * d2_smooth])

    # 新的拟合趋势 = 原趋势 + 高频残差补偿
    trend = p(x) + (res - res_smooth)

    # 4. 应用校正趋势
    for line, trend_fn in corrections.items():
        line_idx = df[line_id_col] == line
        x_vals = df.loc[line_idx, 'rowid'].values.astype(float)
        trend = trend_fn(x_vals)
        df.loc[line_idx, output_ch] = df.loc[line_idx, input_ch] - trend
        if trend_out_ch:
            df.loc[line_idx, trend_out_ch] = trend

    # 写回数据库
    df.drop(columns=['rowid'], errors='ignore').to_sql(
        table, conn, if_exists='replace', index=False)
    conn.close()

    return f"Statistical leveling ({fit_type} Mode) complete for {type_filter} lines. Output: {output_ch}"


def generate_intersection_mask(db_path, input_tab='Tie_intersection', output_tab='Tie_intersection_masked', sigma_threshold=30):
    """
    根据 CROSS_DIFF 的 3-sigma 原则生成 MASK 列。

    参数:
    db_path: 数据库路径
    input_tab: 原始交点表名
    output_tab: 带有 Mask 的输出表名
    sigma_threshold: 离群点阈值，默认 3.0
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {input_tab}", conn)

    # 计算统计信息
    diffs = df['CROSS_DIFF'].dropna()
    mean_val = diffs.mean()
    std_val = diffs.std()

    lower_bound = mean_val - sigma_threshold * std_val
    upper_bound = mean_val + sigma_threshold * std_val

    # 创建 MASK 列：在范围内的设为 1.0，范围外的设为 NaN
    df['MASK'] = 1.0
    outlier_idx = (df['CROSS_DIFF'] < lower_bound) | (
        df['CROSS_DIFF'] > upper_bound)
    df.loc[outlier_idx, 'MASK'] = np.nan

    # 保存为新表
    df.to_sql(output_tab, conn, if_exists='replace', index=False)

    num_outliers = outlier_idx.sum()
    print(f"✅ Mask 表生成完成: {output_tab}")
    print(f"   阈值范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"   识别出离群点: {num_outliers} 个 (总数: {len(df)})")

    conn.close()
    return std_val  # 返回原始 sigma 供后续计算 η


def inport_intersection_mask(db_path, input_tab='Tie_intersection_masked', output_tab='Tie_intersection_masked'):

    conn = sqlite3.connect(db_path)
    mask = pd.read_sql(f"SELECT MASK FROM {input_tab}", conn)
    df = pd.read_sql(f"SELECT * FROM {output_tab}", conn)
    df['MASK'] = mask

    # 保存为新表
    df.to_sql(output_tab, conn, if_exists='replace', index=False)

    conn.close()
    print("导入成功")  # 返回原始 sigma 供后续计算 η
