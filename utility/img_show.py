import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3


def plot_mag_line(db_path, LineID, line_id, x_field='X', y_field='MagCompFilter0_Sunc_IGRF'):
    # 1. 连接数据库并读取该测线数据
    conn = sqlite3.connect(db_path)
    query = f"SELECT {x_field}, {y_field} FROM mag_data WHERE {LineID} = ? ORDER BY {x_field}"
    df = pd.read_sql_query(query, conn, params=(line_id,))
    conn.close()

    # 2. 检查数据
    if df.empty:
        print(f"⚠️ 没有找到测线 {line_id} 的数据")
        return

    # 3. 绘制图像
    plt.figure(figsize=(12, 5))
    plt.plot(df[x_field], df[y_field], color='blue', linewidth=1)
    plt.title(f"Magnetic Anomaly Profile - Line {line_id}")
    plt.xlabel(f"{x_field}")
    plt.ylabel(f"{y_field}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_all_line_ids(db_path, LineID='Line'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT DISTINCT {LineID} FROM mag_data")
    lines = cursor.fetchall()

    print(f"{LineID}：")
    for line in lines:
        print(line[0], end=", ")

    conn.close()


def plot_raw_mag(
    db_path,
    table='mag_data',
    mag_field='MagCompFilter0_Sunc_IGRF',
    x_field='X',
    y_field='Y',
    LineID=None,
    line_id=None,
    cmap='rainbow',
    levels=10,
    title="Raw magnetic anomaly map (all data)"
):
    conn = sqlite3.connect(db_path)

    if line_id:
        query = f"SELECT {x_field}, {y_field}, {mag_field} FROM {table} WHERE {LineID} = ?"
        df = pd.read_sql_query(query, conn, params=(line_id,))
        # title = f"Original magnetic anomaly map ({line_id})"
    else:
        query = f"SELECT {x_field}, {y_field}, {mag_field} FROM {table}"
        df = pd.read_sql_query(query, conn)
        # title = "Raw magnetic anomaly map (all data)"

    conn.close()

    # 清除无效值
    df = df.dropna(subset=[mag_field])
    if df.empty:
        print("No valid magnetic data found.")
        return

    # 自动计算颜色范围
    zmin, zmax = np.percentile(df[mag_field], [5, 95])

    # 绘图
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df[x_field], df[y_field], c=df[mag_field],
                     cmap=cmap, vmin=zmin, vmax=zmax, s=1)
    plt.colorbar(sc, label='Magnetic Anomaly (nT)')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)

    # 可选：插值 + 添加等值线
    if levels > 0:
        from scipy.interpolate import griddata
        xi = np.linspace(df[x_field].min(), df[x_field].max(), 300)
        yi = np.linspace(df[y_field].min(), df[y_field].max(), 300)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((df[x_field], df[y_field]),
                      df[mag_field], (xi, yi), method='linear')
        plt.contour(xi, yi, zi, levels=levels, linewidths=0.3, colors='k')

    plt.show()
