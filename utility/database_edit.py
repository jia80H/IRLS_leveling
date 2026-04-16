import pandas as pd
import sqlite3


# 显示数据库前几行
def show_head(db_path, table, n=5):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {n}", conn)
    conn.close()
    return df
