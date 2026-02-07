import traceback
from typing import Any, Callable, Generic, Self, Sequence, TypeVar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import time
from util import get_project_path, join_path
import abc

OUTPUT_PATH = get_project_path("output/plots/")
PRINT, PLOT = True, True

type Column = str | Sequence[str]

# Clean up data frame before transforming
def _prepare_df(
    df: pd.DataFrame,
    age_col: Column,
    feature_cols: Column,
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

    
def _get_age_format(ages: Sequence, age_scale: int):
    match np.log10(min(age for age in ages if age > 0) * age_scale) // 3:
        case 0: return "ya"
        case 1: return "kya"
        case 2: return "mya"
        case _: return "?ya"

class Dataset:
    
    def __init__(
        self: Self,
        name: str, 
        df: pd.DataFrame,
        feature_cols: dict[Column, str],
        age_format: str,
    ):
        self.name = name
        self.df = df
        self.feature_cols = feature_cols
        self.age_format = age_format
    
    @staticmethod
    def load(
        name: str, 
        df: Callable[[], pd.DataFrame],
        age_col: Column,
        feature_cols: dict[Column, str] | Sequence[Column],
        transform: Callable[[pd.DataFrame, Column, Column], pd.DataFrame] = lambda df, age_col, feature_cols : df, # Transform the input data
        age_scale: int = 1,
    ) -> Self:
        assert len(feature_cols) != 0
        
        if not isinstance(feature_cols, dict):
            feature_cols = {col:col if isinstance(col, str) else " ".join(col) for col in feature_cols}
        
        loaded_df = df()
        loaded_df = transform(loaded_df, age_col, list(feature_cols.keys()))
        loaded_df = _prepare_df(loaded_df, age_col, list(feature_cols.keys()))
        
        age_format = _get_age_format(loaded_df[age_col], age_scale)
        
        loaded_df = loaded_df.set_index(age_col)
        
        return Dataset(
            name=name,
            df=loaded_df,
            feature_cols=feature_cols,
            age_format=age_format
        )
    
    def ages(self: Self) -> np.ndarray:
        return self.df.index.to_numpy(copy=False)
    
    def features(self: Self) -> pd.DataFrame:
        return self.df[self.feature_cols.keys()]
    
    def feature_names(self: Self) -> Sequence[str]:
        return self.feature_cols.values()
    
    def feature_name(self: Self, col: Column) -> str:
        return self.feature_cols[col]
    
    def split(self: Self, features: dict[str, Sequence[Column]]) -> list[Self]:
        return [
            self.__class__(
                name=self.name + " " + feat_name,
                df=self.df[columns],
                feature_cols={col:self.feature_cols[col] for col in columns},
                age_format=self.age_format
            ) for feat_name, columns in features.items()
        ]
        

T = TypeVar('T')

class Model(Generic[T], metaclass = abc.ABCMeta):
    
    def __init__(self: Self, name: str):
        self.name = name
        self.results: dict[Dataset, T] = dict()
        
    @abc.abstractmethod
    def _run(self: Self, data: Dataset):
        pass
    
    @abc.abstractmethod
    def _print(self: Self, dataset: Dataset):
        pass
       
    @abc.abstractmethod 
    def _plot(self: Self, dataset: Dataset) -> Figure:
        pass
    
    def run(self, datasets: list[Dataset]):
        
        print(f"###### Running {self.name} on datasets:")
        for dataset in datasets:
            try:
                print(f"### Running {dataset.name}...")
                start = time.time()
                self.results[dataset] = self._run(dataset)
                print(f"### {self.name} completed in {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))}")
                print()
        
                if PRINT:
                    print(f"###### {self.name} results for {dataset.name}:")
                    try:
                        self._print(dataset)            
                    except Exception:
                        print(f"###### {self.name} results for {dataset.name} failed with exception:")
                        traceback.print_exc()
                    print()
                
                if PLOT:
                    try:
                        fig = self._plot(dataset)
                        fig.savefig(join_path(OUTPUT_PATH, f"{self.name} {dataset.name}.png"))           
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()
            
        print()