import numpy as np
import pandas as pd
import sqlite3
from scipy.interpolate import griddata


def _estimate_line_spacing(df, line_col='line_id', x_col='X', y_col='Y'):
    """估算测线间距"""
    line_centers = df.groupby(line_col).agg(
        {x_col: 'mean', y_col: 'mean'}).reset_index()
    centers = line_centers[[x_col, y_col]].values
    if len(centers) < 2:
        return 50.0
    distances = [np.linalg.norm(centers[i] - centers[i - 1])
                 for i in range(1, len(centers))]
    return np.median(distances) if len(distances) > 0 else 50.0


def fft_grid_prep(
    db_path,
    table_name,
    LineID, X_field, Y_field, channel,
    output_grid,
    grid_cell_size=None,
    xmin=None, ymin=None, xmax=None, ymax=None,
    max_points=5e7  # 限制最大格点数
):
    # Step 1: Load data
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name} WHERE {channel} IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if xmin is None:
        xmin = df[X_field].min()
    if xmax is None:
        xmax = df[X_field].max()
    if ymin is None:
        ymin = df[Y_field].min()
    if ymax is None:
        ymax = df[Y_field].max()

    # Step 2: 自动估算 cell_size
    if grid_cell_size is None:
        spacing = _estimate_line_spacing(
            df, line_col=LineID, x_col=X_field, y_col=Y_field)
        grid_cell_size = spacing / 4.0
        if grid_cell_size <= 0 or np.isnan(grid_cell_size):
            grid_cell_size = 50.0
        print(f"[INFO] 自动估算 cell_size = {grid_cell_size:.2f}")

    # Step 3: 检查格点数
    nx = int((xmax - xmin) / grid_cell_size)
    ny = int((ymax - ymin) / grid_cell_size)
    n_points = nx * ny
    print(f"[INFO] 预期格点数: {nx} × {ny} = {n_points}")

    if n_points > max_points:
        raise MemoryError(f"格点数过大: {n_points}, 请增大 grid_cell_size 或裁剪区域")

    # Step 4: 构建网格
    xi = np.arange(xmin, xmax, grid_cell_size)
    yi = np.arange(ymin, ymax, grid_cell_size)
    xi, yi = np.meshgrid(xi, yi)

    # Step 5: 插值到规则格网
    grid = griddata((df[X_field], df[Y_field]),
                    df[channel], (xi, yi), method="linear")
    mask = np.isnan(grid)
    if np.any(mask):
        grid[mask] = griddata((df[X_field], df[Y_field]),
                              df[channel], (xi, yi), method="nearest")[mask]

    # Step 6: FFT
    fft_data = np.fft.fftshift(np.fft.fft2(grid))
    power_spectrum = np.abs(fft_data)

    # Step 7: 保存结果
    np.savez_compressed(output_grid, x=xi, y=yi, z=grid,
                        fft=fft_data, ps=power_spectrum)
    print(f"[OK] FFT格网及频谱保存至 {output_grid}")
