import sqlite3
import pandas as pd
import numpy as np


def apply_channel_math(db_path, expression: str, table='mag_data'):
    """
    模拟 Oasis Montaj 的 Channel Math 功能。

    示例：
        apply_channel_math(db_path, "Mag_corrected = Mag - noise")
        apply_channel_math(db_path, "logMag = log10(Mag)")
        apply_channel_math(db_path, "Mag_detrend = Mag_corrected - mean(Mag_corrected)")

    支持函数：
        +, -, *, /, **, abs(), log(), log10(), sqrt(), sin(), cos(), mean()

    :param db_path: SQLite 数据库路径
    :param expression: 类似 "new_ch = Mag - noise" 的表达式
    :param table: 表名
    """
    # 拆解表达式
    if '=' not in expression:
        raise ValueError("表达式必须包含赋值号 '='")

    new_col, expr = [s.strip() for s in expression.split('=')]

    # 打开数据库并读取表为 DataFrame
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT rowid, * FROM {table}", conn)

    # 构造可用的安全数学环境
    safe_dict = {
        'np': np,
        'abs': np.abs,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'mean': lambda x: np.mean(x),
    }
    # 添加所有字段名为变量
    for col in df.columns:
        if col not in safe_dict:
            safe_dict[col] = df[col].values

    # 计算表达式
    try:
        result = eval(expr, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        raise RuntimeError(f"表达式执行出错: {e}")

    # 检查新字段是否已存在，若无则添加
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {new_col} REAL")
    except sqlite3.OperationalError:
        pass  # 字段可能已存在

    # 更新每一行
    for rowid, val in zip(df['rowid'], result):
        conn.execute(
            f"UPDATE {table} SET {new_col} = ? WHERE rowid = ?", (float(val), int(rowid)))

    conn.commit()
    conn.close()
    print(f"通道计算完成：{new_col} 写入数据库 {table}")


def split_survery_ties(db_path, table='mag_data', LineID='Line', x='X', y='Y'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    survey = df[df.Line.str.startswith("L")].copy()
    tie = df[df.Line.str.startswith("T")].copy()

    conn.close()
    return survey, tie


class MagSurveyData:
    def __init__(self, db_path, table_name, line_col="Line", x_col="x", y_col="y", mag_col="Mag"):
        self.db_path = db_path
        self.table_name = table_name
        self.line_col = line_col
        self.x_col = x_col
        self.y_col = y_col
        self.mag_col = mag_col
        self.df = None
        self.survey_lines = {}
        self.tie_lines = {}

    def load_data(self):
        """从 SQLite3 加载数据"""
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT {self.line_col}, {self.x_col}, {self.y_col}, {self.mag_col} FROM {self.table_name} WHERE {self.mag_col} IS NOT NULL"
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"数据加载完成，共 {len(self.df)} 条记录")

    def split_lines(self):
        """按 Line 类型拆分测线/切割线"""
        if self.df is None:
            raise ValueError("请先调用 load_data()")

        survey_df = self.df[self.df[self.line_col].str.startswith("L")]
        tie_df = self.df[self.df[self.line_col].str.startswith("T")]

        # 存储为 dict {line_id: DataFrame}
        self.survey_lines = {lid: g for lid,
                             g in survey_df.groupby(self.line_col)}
        self.tie_lines = {lid: g for lid, g in tie_df.groupby(self.line_col)}

        print(f"识别到 {len(self.survey_lines)} 条测线, {len(self.tie_lines)} 条切割线")

    def get_line(self, line_id):
        """获取指定航线的数据 DataFrame"""
        if line_id in self.survey_lines:
            return self.survey_lines[line_id]
        elif line_id in self.tie_lines:
            return self.tie_lines[line_id]
        else:
            raise KeyError(f"未找到航线 {line_id}")

    def summary(self):
        """输出数据统计信息"""
        if self.df is None:
            return "数据未加载"
        return self.df.describe(include="all")


class MagSurveyData2:
    def __init__(self, db_path, table_name,
                 line_col="Line", x_col="x", y_col="y", mag_col="Mag"):
        self.db_path = db_path
        self.table_name = table_name
        self.line_col = line_col
        self.x_col = x_col
        self.y_col = y_col
        self.mag_col = mag_col
        self.df = None
        self.survey_lines = {}
        self.tie_lines = {}
        self.has_line = True   # 标记是否真的存在航线列

    def load_data(self):
        """从 SQLite3 加载数据"""
        conn = sqlite3.connect(self.db_path)

        # 检查表结构是否包含 line_col
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        cols = [row[1] for row in cursor.fetchall()]
        if self.line_col not in cols:
            print(f"⚠️ 表中没有列 {self.line_col}，将自动生成虚拟航线 ID")
            self.has_line = False
            query = f"SELECT {self.x_col}, {self.y_col}, {self.mag_col} FROM {self.table_name} WHERE {self.mag_col} IS NOT NULL"
        else:
            query = f"SELECT {self.line_col}, {self.x_col}, {self.y_col}, {self.mag_col} FROM {self.table_name} WHERE {self.mag_col} IS NOT NULL"

        self.df = pd.read_sql_query(query, conn)
        conn.close()

        # 如果没有航线列，生成一个虚拟列
        if not self.has_line:
            self.df[self.line_col] = "LINE_1"

        print(f"✅ 数据加载完成，共 {len(self.df)} 条记录")

    def split_lines(self):
        """按 Line 类型拆分测线/切割线"""
        if self.df is None:
            raise ValueError("请先调用 load_data()")

        if not self.has_line:
            # 没有航线列的情况 → 只有一条虚拟测线
            self.survey_lines = {"LINE_1": self.df.copy()}
            self.tie_lines = {}
            print("⚠️ 无航线列，已生成 1 条虚拟测线 (LINE_1)")
            return

        # 正常情况：按 L/T 前缀拆分
        survey_df = self.df[self.df[self.line_col].astype(
            str).str.startswith("L")]
        tie_df = self.df[self.df[self.line_col].astype(
            str).str.startswith("T")]

        self.survey_lines = {lid: g for lid,
                             g in survey_df.groupby(self.line_col)}
        self.tie_lines = {lid: g for lid, g in tie_df.groupby(self.line_col)}

        print(f"识别到 {len(self.survey_lines)} 条测线, {len(self.tie_lines)} 条切割线")

    def get_line(self, line_id):
        """获取指定航线的数据 DataFrame"""
        if line_id in self.survey_lines:
            return self.survey_lines[line_id]
        elif line_id in self.tie_lines:
            return self.tie_lines[line_id]
        else:
            raise KeyError(f"未找到航线 {line_id}")

    def summary(self):
        """输出数据统计信息"""
        if self.df is None:
            return "数据未加载"
        return self.df.describe(include="all")
