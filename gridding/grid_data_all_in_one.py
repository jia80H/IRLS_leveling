from .bidirectional_gridding import bidirectional_gridding
from .minimum_curvature_gridding import minimum_curvature_gridding
from .idw_gridding import idw_gridding
from .fft_grid import fft_grid_prep


def grid_data(
    survey,                 # MagSurveyData 实例
    # 可选: "bidirectional" / "minimum_curvature" / "idw" / "fft"
    method="minimum_curvature",
    output_grid="grid_out",  # 输出文件名（npz）
    grid_cell_size=None,
    xmin=None, xmax=None, ymin=None, ymax=None,
    **kwargs                # 传给具体方法的参数
):
    """
    通用格网化接口
    """
    if method == "bidirectional":
        return bidirectional_gridding(
            survey.db_path, survey.table_name,
            survey.line_col, survey.x_col, survey.y_col, survey.mag_col,
            output_grid,
            grid_cell_size=grid_cell_size,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            **kwargs
        )

    elif method == "minimum_curvature":
        return minimum_curvature_gridding(
            survey.db_path, survey.table_name,
            survey.line_col, survey.x_col, survey.y_col, survey.mag_col,
            output_grid,
            grid_cell_size=grid_cell_size,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            **kwargs
        )

    elif method == "idw":
        return idw_gridding(
            survey.db_path, survey.table_name,
            survey.line_col, survey.x_col, survey.y_col, survey.mag_col,
            output_grid,
            grid_cell_size=grid_cell_size,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            **kwargs
        )

    elif method == "fft":
        return fft_grid_prep(
            survey.db_path, survey.table_name,
            survey.line_col, survey.x_col, survey.y_col, survey.mag_col,
            output_grid,
            grid_cell_size=grid_cell_size,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            **kwargs
        )
    else:
        raise ValueError(f"未知格网方法: {method}")
