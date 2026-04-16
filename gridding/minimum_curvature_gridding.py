import numpy as np
import pandas as pd
import sqlite3
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


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


def minimum_curvature_gridding(
    db_path,
    table_name,
    LineID, X_field, Y_field, channel,
    output_grid,
    grid_cell_size=None,
    xmin=None, ymin=None, xmax=None, ymax=None,
    log_option="linear",
    log_minimum=1.0,
    lowpass_factor=1,
    blanking_distance=None,
    tolerance=0.01,
    percent_pass_tolerance=95,
    max_iterations=100,
    coarse_grid=1,
    search_radius=None,
    internal_tension=0.5,
    extend_cells=0,
    weighting_power=2,
    weighting_slope=0.0,
):
    # Step 1: Load data
    conn = sqlite3.connect(db_path)
    query = f"SELECT {LineID}, {X_field}, {Y_field}, {channel} FROM {table_name} WHERE {channel} IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if xmin is None or ymin is None or xmax is None or ymax is None:
        xmin = df[X_field].min() if xmin is None else xmin
        xmax = df[X_field].max() if xmax is None else xmax
        ymin = df[Y_field].min() if ymin is None else ymin
        ymax = df[Y_field].max() if ymax is None else ymax
    # Apply bounding box
    df = df[(df[X_field] >= xmin) & (df[X_field] <= xmax) &
            (df[Y_field] >= ymin) & (df[Y_field] <= ymax)]

    if df.empty:
        raise ValueError("No data found in specified bounding box.")

    # Handle log options
    if "log" in log_option:
        df[channel] = df[channel].clip(lower=log_minimum)
        df[channel] = np.log(df[channel])
        log_applied = True
    else:
        log_applied = False

    # Prepare grid
    if grid_cell_size is None:
        line_spacing = _estimate_line_spacing(
            df, line_col=LineID, x_col=X_field, y_col=Y_field)
        print(f"测线间距：{line_spacing}")
        grid_cell_size = line_spacing / 4.0

    xi = np.arange(xmin, xmax, grid_cell_size)
    yi = np.arange(ymin, ymax, grid_cell_size)
    xi, yi = np.meshgrid(xi, yi)

    # Step 2: Initial interpolation using 'linear' or 'nearest' (fallback)
    grid = griddata((df[X_field], df[Y_field]),
                    df[channel], (xi, yi), method='linear')
    mask = np.isnan(grid)
    if np.any(mask):
        grid[mask] = griddata((df[X_field], df[Y_field]), df[channel],
                              (xi, yi), method='nearest')[mask]

    # Optional low-pass filtering (simple Gaussian for now)
    if lowpass_factor > 1:
        grid = gaussian_filter(grid, sigma=lowpass_factor)

    # If log needs to be saved as linear, take exp
    if log_applied and "save as linear" in log_option:
        grid = np.exp(grid)

    # Save to output
    np.savez_compressed(output_grid, x=xi, y=yi, z=grid)
