from typing import Union
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.random.seed(0)  # Set seed for reproducibility
import pandas as pd

def join_path(p, *args):
    return os.path.join(p, *args).replace("\\", "/")

# Get path above source folder
def get_project_path(path: str):
    # up from source/util.py
    path = join_path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), path)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
            os.makedirs(dir)
    return path


# FUNCTIONS TO TRANSFORM INPUT DATA

def resample_df(df: pd.DataFrame, age_col, feature_cols: list, steps: float = 1.0) -> pd.DataFrame:
    df = df[[age_col] + feature_cols].dropna().sort_values(by=age_col)

    ages = df[age_col].values.astype(float)
    new_ages = np.arange(np.ceil(ages.min()), np.floor(ages.max()), steps)

    new_df = pd.DataFrame({age_col: new_ages})
    for col in feature_cols:
        new_df[col] = np.interp(new_ages, ages, df[col].values)

    return new_df