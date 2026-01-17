from typing import Any, Callable, Union
import pandas as pd
import matplotlib.pyplot as plt
import time
from util import get_project_path, join_path

OUTPUT_PATH = get_project_path("../output/plots/")
PRINT, PLOT = True, True

__PlotRunner = Callable[[str, str], None]
__ModelRunner = tuple[str, Callable[[list, list], tuple[__PlotRunner, __PlotRunner]]]

# TODO: rename models argument to something that can include both model and metric based indicator finders

def run_models_on_data(
    name: str, # Name of data
    df: Callable[[], pd.DataFrame], # Function to load data
    age_col: Any, # Age columns
    feature_cols: Any, # Feature columns (Columns we will run the model on) (TODO use the correct name for these variables)
    models: Union[__ModelRunner, list[__ModelRunner]], # Model runner(s) we will use
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
    
    def run_model(runner: __ModelRunner) -> Any:
        model_name, runner = runner
        
        print(f"###### Running {model_name} for {name}")
        
        start = time.time()
        # Run the user-specified model(s) given to the parent function
        print_results, plot_results = runner(df[age_col].values, df[feature_cols].values)
        print(f"###### {model_name} for {name} completed in {time.time() - start:.2f}s")        
        print()
        
        if PRINT:
            print(f"###### Retrieved {model_name} results for {name}")
            print_results(name, model_name, age_format)
            print()
        
        if PLOT:
            plt.figure(figsize=(12,4))
            plt.xlabel(f"Age ({age_format})")
            plot_results(name, model_name, age_format)
            plt.legend()
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(join_path(OUTPUT_PATH, f"{name} {model_name}.png"))
            plt.close()
            
        print()
    
    if isinstance(models, list):
        for m in models:
            run_model(m)
    else:
        run_model(models)