from typing import Any, Callable, Union
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import time
from util import get_project_path, join_path
import abc

OUTPUT_PATH = get_project_path("output/plots/")
PRINT, PLOT = True, True

class Dataset:
    
    def __init__(
        self, 
        name: str, 
        df: Callable[[], pd.DataFrame],
        age_col: Any,
        feature_cols: list,
        transform: Callable[[pd.DataFrame, Any, Any], pd.DataFrame] = lambda df, age_col, feature_cols : df, # Transform the input data
        age_scale: int = 1,
    ):
        self.name = name
        
        # Clean up data frame before transforming
        def prepare_df(
            df: pd.DataFrame,
            age_col,
            feature_cols: list,
        ) -> pd.DataFrame:
            
            # Select relevant columns and ensure age is present
            df_sel = df[[age_col] + feature_cols].sort_values(by=age_col).reset_index(drop=True)
            df_sel = df_sel.dropna(subset=[age_col])

            if df_sel.empty:
                raise ValueError(f"No rows remain after dropping rows with missing '{age_col}'. Check your data.")

            # If all feature values are NaN for every row, abort with helpful message
            if df_sel[feature_cols].dropna(how='all').empty:
                raise ValueError(
                    "No feature values found for any of the requested feature columns. "
                    "This may be caused by incorrect feature column names or a pivot that produced only NaNs. "
                    f"Requested features: {feature_cols}"
                )

            try:
                df_sel[feature_cols] = df_sel[feature_cols].astype(float)
            except Exception:
                # If conversion fails, attempt coercion to numeric (non-numeric -> NaN)
                for c in feature_cols:
                    df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")

            # Final drop: remove rows that are still entirely NaN in the feature columns
            df_sel = df_sel.dropna(subset=feature_cols, how='all')

            if df_sel.empty:
                raise ValueError("No rows with usable feature values after imputation. Check your data and feature selection.")

            return df_sel
        
        self.df = df()
        self.df = transform(self.df, age_col, feature_cols)
        self.df = prepare_df(self.df, age_col, feature_cols)
        if age_scale != 1:
            self.df[age_col] = self.df[age_col].multiply(age_scale)
        
        self.age_col = age_col
        self.feature_cols = feature_cols
    
    def ages(self):
        return self.df[self.age_col].values
        
    def features(self):
        return self.df[self.feature_cols].values

class Model(metaclass = abc.ABCMeta):
    
    def __init__(self, name: str):
        self.name = name
        
    @abc.abstractmethod
    def runner(self, data: list[Dataset]) -> tuple[Callable[[], None], Callable[[], list[Figure]]]:
        pass
    
    def run(self, data: list[Dataset]):
        
        print(f"###### Running {self.name} on datasets:")
        for d in data:
            print(f"### {d.name}")
        
        start = time.time()
        # Run the user-specified model(s) given to the parent function
        print_results, plot_results = self.runner(data)
        print(f"###### {self.name} completed in {time.time() - start:.2f}s")        
        print()
        
        if PRINT:
            print(f"###### Retrieved results for {self.name}:")
            print_results()
            print()
        
        if PLOT:
            fig = plot_results()
            fig.savefig(join_path(OUTPUT_PATH, f"{self.name} {[d.name.split(" ")[0] for d in data]}.png"))
            
        print()