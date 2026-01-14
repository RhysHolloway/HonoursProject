from typing import Any, Callable, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "../output")
PRINT, PLOT = True, True

def filter_points(points, ages, scores, min_distance: Union[float, int]):

    # Sort peaks by descending height (strongest first)
    order = np.argsort(-scores)
    sorted_peaks = points[order]

    kept = []
    used = np.zeros(len(sorted_peaks), dtype=bool)

    for i, p in enumerate(sorted_peaks):
        if used[i]:
            continue  # already removed due to being too close to a better peak

        # keep this peak
        kept.append(p)

        # mask out all peaks within min_years
        age_p = ages[p]
        for j in range(i + 1, len(sorted_peaks)):
            if used[j]:
                continue
            p2 = sorted_peaks[j]
            if abs(ages[p2] - age_p) < min_distance:
                used[j] = True

    # return sorted in chronological order
    kept = np.array(kept, dtype=int)
    return kept[np.argsort(ages[kept])]

def resample_df(df, age_col, feature_cols: list, steps: float = 1.0):
    df = df[[age_col] + feature_cols].dropna().sort_values(by=age_col)

    ages = df[age_col].values.astype(float)
    new_ages = np.arange(np.ceil(ages.min()), np.floor(ages.max()), steps)

    new_df = pd.DataFrame({age_col: new_ages})
    for col in feature_cols:
        new_df[col] = np.interp(new_ages, ages, df[col].values.astype(float))

    return new_df

PlotRunner = Callable[[str], None]
ModelRunner = tuple[str, Callable[[list, list], tuple[PlotRunner, PlotRunner]]]

def run_models_on_data(
    name: str,
    df: Callable[[], pd.DataFrame],
    age_col: Any,
    feature_cols: Any,
    models: Union[ModelRunner, list[ModelRunner]],
    transform: Callable[[pd.DataFrame, Any, Any], pd.DataFrame] = lambda df, age_col, feature_cols : df,
    age_format: str = "Mya",
) -> Any:
    
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
    
    df = prepare_df(df(), age_col, feature_cols)
    df = transform(df, age_col, feature_cols)
    
    ages = df[age_col].values
    features = df[feature_cols].values
    
    def run_model(runner: ModelRunner) -> Any:
        model_name, runner = runner
        
        print(f"###### Running {model_name} for {name}")
        start = time.time()
        print_results, plot_results = runner(ages, features)
        print(f"###### {model_name} for {name} completed in {time.time() - start:.2f}s")
        if PRINT:
            print(f"###### Retrieved {model_name} results for {name}")
            print_results(age_format)
        print()
        
        if PLOT:
            plt.figure(figsize=(12,4))
            plot_results(age_format)
            plt.xlabel(f"Age ({age_format})")
            plt.legend()
            plt.title(f"Tipping points for {name} using {model_name}", pad=80)
            plt.gca().invert_xaxis()
            plt.tight_layout()
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            plt.savefig(os.path.join(OUTPUT_PATH, f"{name} {model_name}.png"))
            plt.close()
    
    if isinstance(models, list):
        return [run_model(m) for m in models]
    else:
        return run_model(models)