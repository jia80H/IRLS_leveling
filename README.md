# IRLS_leveling

A robust aeromagnetic data leveling method using Iteratively Reweighted Least Squares (IRLS) with Huber loss for tie line adjustment.

## Overview

This project implements a robust leveling algorithm for aeromagnetic survey data that uses **IRLS with Huber loss** instead of traditional ordinary least squares (OLS). The method is particularly effective at handling outliers in intersection errors between tie lines and traverse lines.

### Key Features

- **Robust Estimation**: Uses Huber loss function to automatically down-weight outliers during polynomial fitting
- **Two-Stage Leveling**: Processes tie lines first, then traverse lines
- **Mask Support**: Optional sigma-based masking to exclude severe outliers
- **Multiple Interpolation Methods**: Bidirectional and minimum curvature gridding
- **Comprehensive Evaluation**: SCI-level metrics including standard deviation, MAE

## Directory Structure

```
IRLS_leveling/
├── main.ipynb              # Main demonstration notebook
├── README.md               # This file
├── 00ge_data/              # Data directory
│   └── *.npz               # Grid files
├── database/               # Database utilities
│   └── process_aeromag_csv.py
├── evaluation/             # Evaluation metrics
│   └── evaluate.py
├── gridding/               # Gridding methods
│   ├── bidirectional_gridding.py
│   ├── minimum_curvature_gridding.py
│   └── view_grd_file.py
├── leveling/               # Core leveling algorithms
│   ├── irls_huber.py       # IRLS robust fitting
│   └── lev_tie_line.py     # Tie line leveling pipeline
├── real_data/              # Example data
│   └── leveling/example.csv
└── utility/                # Utility functions
    ├── database_edit.py
    ├── general_utility.py
    └── img_show.py
```

## Methodology

### Traditional vs. Robust Leveling

| Aspect                | Traditional (OLS) | IRLS-Huber                    |
| --------------------- | ----------------- | ----------------------------- |
| Outlier Handling      | Sensitive         | Robust                        |
| Weighting             | Equal weights     | Iteratively reweighted        |
| Residual Distribution | Assumes Gaussian  | Adapts to outliers            |
| Convergence           | Single fit        | Iterative (max 50 iterations) |

### Pipeline

1. **Intersection Calculation**: Compute intersection points between tie lines (T*) and traverse lines (L*)
2. **Mask Generation** (optional): Exclude intersections exceeding N×sigma threshold
3. **Tie Line Leveling**: Fit polynomial trends to tie lines using IRLS-Huber
4. **Load Corrections**: Apply computed corrections to all data points
5. **Traverse Line Leveling**: Fit polynomial trends to traverse lines using IRLS-Huber
6. **Final Evaluation**: Calculate SCI metrics at intersections


## Installation

```bash
pip install numpy pandas scipy shapely matplotlib
```

## Quick Start

```python
from leveling.lev_tie_line import (
    tieline_intersection,
    load_correction,
    statistical_level,
    generate_intersection_mask
)
from evaluation.evaluate import calculate_metrics_for_sci

# 1. Calculate intersections
tieline_intersection(
    db_path='./00ge_data/example.db',
    LineID='Line', xch='x', ych='y', data_ch='Mag',
    output_tab='Tie_intersection'
)

# 2. Generate mask (optional)
generate_intersection_mask(
    db_path='./00ge_data/example.db',
    input_tab='Tie_intersection',
    output_tab='Tie_intersection_masked',
    sigma_threshold=3
)

# 3. Load corrections for tie lines
load_correction(
    db_path='./00ge_data/example.db',
    intersection_table='Tie_intersection',
    main_table='mag_data',
    line_id_col='Line',
    mask_channel='MASK',
    process_line_types='TIE'
)

# 4. Statistical leveling with robust mode
statistical_level(
    db_path='./00ge_data/example.db',
    table='mag_data',
    line_id_col='Line',
    type_filter='TIE',
    input_ch='Mag',
    output_ch='LEVELLED_IRLS',
    trend_order=1,
    robust_mode=True  # Enable IRLS-Huber
)

# 5. Repeat for traverse lines (LINE)
# ... (see main.ipynb for complete pipeline)

# 6. Evaluate results
eval_result = calculate_metrics_for_sci(
    db_path, 'Final_intersection', 'Tie_intersection_masked'
)
```

## Evaluation Metrics

The project uses SCI (Scientific) level metrics:

| Metric                    | Description                                  |
| ------------------------- | -------------------------------------------- |
| **Std. Dev (σ)**   | Standard deviation of intersection residuals |
| **MAE**             | Mean Absolute Error                          |
| **Improvement (%)** | Reduction in σ relative to baseline         |

## Example Results

See `main.ipynb` for complete benchmarking results comparing:

- **Raw**: Original unleveled data
- **STA**: Traditional statistical leveling (OLS)
- **IRLS**: Robust leveling with Huber loss
- **With Mask**: Using sigma-based outlier masking

## Core Functions

### leveling/irls_huber.py

- `robust_polynomial_fit()`: IRLS with Huber/Tukey weight functions

### leveling/lev_tie_line.py

- `tieline_intersection()`: Calculate line intersections
- `generate_intersection_mask()`: Create outlier masks
- `load_correction()`: Apply intersection corrections
- `statistical_level()`: Polynomial trend fitting (OLS or robust)
- `calculate_metrics_for_sci()`: Compute evaluation metrics

### gridding/

- `bidirectional_gridding()`: Interpolate along and across lines
- `minimum_curvature_gridding()`: Minimum curvature surface fitting

## License

MIT License


## Contact

For questions and issues, please open an issue on the project repository.
