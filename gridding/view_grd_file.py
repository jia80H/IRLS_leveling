import numpy as np
import matplotlib.pyplot as plt


def view_grd_file(grd_path, cmap='rainbow', title="Gridded Magnetic Anomaly", levels=20):
    data = np.load(grd_path)
    xi = data['x']
    yi = data['y']
    zi = data['z']

    # 自动设置颜色范围增强对比度
    zmin, zmax = np.percentile(zi[~np.isnan(zi)], [5, 95])

    plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(xi, yi, zi, shading='auto',
                       cmap=cmap, vmin=zmin, vmax=zmax)
    plt.colorbar(c, label='Magnetic Anomaly (nT)')
    plt.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def view_grd_diff(grd_path1, grd_path2, cmap='rainbow', title="Gridded Magnetic Anomaly", levels=20):
    data1 = np.load(grd_path1)
    xi = data1['x']
    yi = data1['y']
    zi = data1['z']

    data2 = np.load(grd_path2)
    zi = zi - data2['z']

    # 自动设置颜色范围增强对比度
    zmin, zmax = np.percentile(zi[~np.isnan(zi)], [5, 95])

    plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(xi, yi, zi, shading='auto',
                       cmap=cmap, vmin=zmin, vmax=zmax)
    plt.colorbar(c, label='Magnetic Anomaly (nT)')
    plt.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def compare_grids(grd_paths, labels=None, cmap='rainbow', levels=20, figsize=(14, 6)):
    """
    并排对比多个格网结果（适合调平前后、不同方法效果对比）

    参数:
        grd_paths : list[str]   # npz 格网文件路径列表
        labels    : list[str]   # 每个格网对应标题
        cmap      : str         # 颜色映射
        levels    : int         # 等值线数量
        figsize   : tuple       # 图像尺寸
    """
    n = len(grd_paths)
    if labels is None:
        labels = [f"Grid {i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, grd_path, label in zip(axes, grd_paths, labels):
        data = np.load(grd_path)
        xi, yi, zi = data['x'], data['y'], data['z']

        # 自动设置颜色范围增强对比度
        zmin, zmax = np.percentile(zi[~np.isnan(zi)], [5, 95])

        c = ax.pcolormesh(xi, yi, zi, shading='auto',
                          cmap=cmap, vmin=zmin, vmax=zmax)
        ax.contour(xi, yi, zi, levels=levels, colors='black', linewidths=0.3)
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.grid(True)

    # 在最后统一加 colorbar
    fig.colorbar(c, ax=axes, orientation="vertical",
                 label="Magnetic Anomaly (nT)")
    plt.show()
