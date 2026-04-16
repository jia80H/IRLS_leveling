import numpy as np
import pandas as pd
import sqlite3
from scipy.interpolate import Akima1DInterpolator, CubicSpline, interp1d, griddata
from scipy.ndimage import gaussian_filter


def _estimate_spacing(df, x_col, y_col):
    coords = df[[x_col, y_col]].values
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.median(diffs)


def _estimate_line_spacing(df, line_col='line_id', x_col='X', y_col='Y'):
    # 提取每条测线的平均位置
    line_centers = df.groupby(line_col).agg(
        {x_col: 'mean', y_col: 'mean'}).reset_index()

    # 计算测线中心点之间的欧氏距离
    centers = line_centers[[x_col, y_col]].values
    distances = []
    for i in range(1, len(centers)):
        d = np.linalg.norm(centers[i] - centers[i-1])
        distances.append(d)

    # 返回测线间平均距离（去掉异常值）
    return np.median(distances)


def _get_interpolator(method, x, y):
    if method == 'Akima':
        return Akima1DInterpolator(x, y)
    elif method == 'cubic':
        return CubicSpline(x, y)
    elif method == 'linear':
        return interp1d(x, y, kind='linear', bounds_error=False)
    elif method == 'nearest':
        return interp1d(x, y, kind='nearest', bounds_error=False)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def bidirectional_gridding(
    db_path,
    table_name,
    LineID, X_field, Y_field, channel,
    output_grid,
    grid_cell_size=None,
    xmin=None, ymin=None, xmax=None, ymax=None,
    max_line_sep=None,
    max_point_sep=None,
    cells_extend=1,
    down_interp='Akima',
    cross_interp='Akima',
    lowpass_wl=None,
    highpass_wl=None,
    nonlinear_tol=None,
    prefilter_step=None,
    presort='none',
    log_option='linear',
    log_minimum=1.0,
    trend_angle=0.0,
    force_grid='default',
    has_gradient=False
):
    # Step 1: Load Data
    conn = sqlite3.connect(db_path)
    query = f"SELECT {LineID}, {X_field}, {Y_field}, {channel} FROM {table_name} WHERE {channel} IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if presort == 'pre-sort data':
        df.sort_values([LineID, Y_field], inplace=True)
    elif presort == 'remove backtracks':
        df = df.drop_duplicates(subset=[X_field, Y_field])

    if xmin is None:
        xmin = df[X_field].min()
    if xmax is None:
        xmax = df[X_field].max()
    if ymin is None:
        ymin = df[Y_field].min()
    if ymax is None:
        ymax = df[Y_field].max()
    df = df[(df[X_field] >= xmin) & (df[X_field] <= xmax) &
            (df[Y_field] >= ymin) & (df[Y_field] <= ymax)]

    # Step 2: Log options
    if 'log' in log_option:
        df[channel] = df[channel].clip(lower=log_minimum)
        df[channel] = np.log(df[channel])
        log_applied = True
    else:
        log_applied = False

    # Step 3: Estimate cell size if not given
    if grid_cell_size is None:
        spacing = _estimate_line_spacing(df, LineID, X_field, Y_field)
        grid_cell_size = spacing / 4.0
        print(f"Estimated grid cell size: {grid_cell_size}")

    # Step 4: Prepare grid
    xi = np.arange(xmin, xmax + grid_cell_size, grid_cell_size)
    yi = np.arange(ymin, ymax + grid_cell_size, grid_cell_size)
    xi, yi = np.meshgrid(xi, yi)

    # Step 5: Interpolate along line direction
    line_grids = []
    for line_id, group in df.groupby(LineID):
        x = group[X_field].values
        y = group[Y_field].values
        z = group[channel].values

        if len(z) < 4:
            continue
        try:
            sort_idx = np.argsort(y)
            interp = _get_interpolator(down_interp, y[sort_idx], z[sort_idx])
            new_z = interp(yi[:, 0])
            line_grids.append(
                (np.full_like(yi[:, 0], np.mean(x)), yi[:, 0], new_z))
        except Exception:
            sort_idx = np.argsort(x)
            interp = _get_interpolator(down_interp, x[sort_idx], z[sort_idx])
            new_z = interp(xi[0, :])
            line_grids.append(
                (xi[0, :], np.full_like(xi[0, :], np.mean(y)), new_z))
            continue

    if not line_grids:
        raise ValueError("没有有效测线用于插值")

    # Combine down-line interpolated results
    all_x = np.concatenate([g[0] for g in line_grids])
    all_y = np.concatenate([g[1] for g in line_grids])
    all_z = np.concatenate([g[2] for g in line_grids])

    # Step 6: Across-line interpolation
    grid = griddata((all_x, all_y), all_z, (xi, yi), method='linear')
    mask = np.isnan(grid)
    if np.any(mask):
        grid[mask] = griddata((all_x, all_y), all_z,
                              (xi, yi), method='nearest')[mask]

    # Step 7: Optional low/high-pass filtering
    if lowpass_wl:
        sigma = lowpass_wl / (2 * grid_cell_size)
        grid = gaussian_filter(grid, sigma=sigma)
    if highpass_wl:
        sigma = highpass_wl / (2 * grid_cell_size)
        low = gaussian_filter(grid, sigma=sigma)
        grid = grid - low

    # Step 8: Restore log if needed
    if log_applied and 'save as linear' in log_option:
        grid = np.exp(grid)

    # Step 9: Save grid
    np.savez_compressed(output_grid, x=xi, y=yi, z=grid)
    print(f"网格保存至 {output_grid}")
