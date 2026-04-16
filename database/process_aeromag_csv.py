import pandas as pd
import sqlite3
import os


def process_aeromag_csv(
    csv_path: str,
    db_path: str = './00ge_data/aeromag.db',
    export_per_line: bool = False,
    export_folder: str = 'split_lines',
    header: int = 0,
    sep: str = ',',
):
    # 1. 读取 CSV 文件
    print(f"📂 正在读取文件：{csv_path}")
    df = pd.read_csv(csv_path, header=header, sep=sep)

    # 2. 规范字段名
    rename_map = {
        'Line.1': 'LineID',
        # 'X': 'Easting',
        # 'Y': 'Northing',
        '__Date': 'Date',
        'Time': 'UTCTime'
    }
    df.rename(columns={k: v for k, v in rename_map.items()
              if k in df.columns}, inplace=True)

    # 3. 构造标准时间列（可选）
    try:
        df['DateTime'] = pd.to_datetime(
            df['Date'].astype(str) + df['UTCTime'].astype(str),
            format="%Y%m%d%H%M%S.%f", errors='coerce'
        )
    except Exception as e:
        print(f"⚠️ 时间列构造失败：{e}")

    # 4. 写入 SQLite 主表
    print(f"🗄️ 正在写入 SQLite 数据库：{db_path}")
    conn = sqlite3.connect(db_path)
    df.to_sql('mag_data', conn, if_exists='replace', index=False)
    conn.close()
    print("✅ 数据成功写入主表 mag_data")

    # 5. 可选：按测线拆分 CSV 文件
    if export_per_line:
        os.makedirs(export_folder, exist_ok=True)
        print(f"📤 正在按测线导出到目录：{export_folder}")
        for line_id in df['LineID'].unique():
            sub_df = df[df['LineID'] == line_id]
            filename = os.path.join(export_folder, f"{line_id}.csv")
            sub_df.to_csv(filename, index=False)
        print("✅ 测线文件导出完成")
