from typing import Any, Callable, Union
import pandas as pd
import matplotlib.pyplot as plt
import time
from util import get_project_path, join_path

OUTPUT_PATH = get_project_path("output/plots/")
PRINT, PLOT = True, True

__Runner = tuple[str, Callable[[list, list], tuple[Callable[[str, str], None], Callable[[plt.Figure, str, str], None]]]]

def run_analyses_on_data(
    name: str, # Name of dataset
    df: Callable[[], pd.DataFrame], # Function to load dataset
    age_col: Any, # Age columns
    feature_cols: Any, # Feature columns (Columns we will run the model on)
    analyses: Union[__Runner, list[__Runner]], # Model runner(s) we will use
    transform: Callable[[pd.DataFrame, Any, Any], pd.DataFrame] = lambda df, age_col, feature_cols : df, # Transform the input data
    age_format: str = "Mya", # Age format to display in output
) -> Any:
    
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
    
    df = df()
    df = transform(df, age_col, feature_cols)
    df = prepare_df(df, age_col, feature_cols)
    
    def run_analysis(runner: __Runner) -> Any:
        runner_name, runner = runner
        data_name = name
        
        print(f"###### Running {runner_name} for {data_name}")
        
        start = time.time()
        # Run the user-specified model(s) given to the parent function
        print_results, plot_results = runner(df[age_col].values, df[feature_cols].values)
        print(f"###### {runner_name} for {data_name} completed in {time.time() - start:.2f}s")        
        print()
        
        if PRINT:
            print(f"###### Retrieved {runner_name} results for {data_name}")
            print_results(data_name, age_format)
            print()
        
        if PLOT:
            fig = plt.figure(figsize=(12,4))
            fig.tight_layout()
            plot_results(fig, data_name, age_format)
            fig.savefig(join_path(OUTPUT_PATH, f"{data_name} {runner_name}.png"))
            
        print()
    
    if isinstance(analyses, list):
        for a in analyses:
            run_analysis(a)
    else:
        run_analysis(analyses)