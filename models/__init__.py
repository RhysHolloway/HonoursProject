from __future__ import annotations

import traceback
from typing import Any, Callable, Generic, Iterable, Literal, Self, Sequence, TypeVar, ValuesView
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import abc

type Column = str | Sequence[str]
type FeatureColumns = dict[Column, str] | Sequence[Column]

def compute_residuals(data: pd.Series, span: float = 0.2, type: Literal["Gaussian", "Lowess"] = "Lowess") -> pd.Series:
    from scipy.ndimage import gaussian_filter
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothing: np.ndarray
    data = pd.Series(data, index=data.index.get_level_values("time"))
    match type:
        case "Gaussian":
            # Standard deviation of kernel given bandwidth
            # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
            span = span * len(data) if 0 < span <= 1 else span
            smoothing = gaussian_filter(data, sigma=(0.25 / 0.675) * span, mode="reflect")
        case "Lowess":
            span = span if 0 < span <= 1 else span / len(data)
            smoothing = lowess(data.to_numpy(), data.index.to_numpy(), frac=span, return_sorted=False)
        case _:
            raise ValueError(f"Invalid detrending type: {type}")
    return pd.Series(data.to_numpy() - smoothing, index=data.index, name="residuals")

def space_indices(series: pd.Series, spacing: int) -> list[int]:
    first = series.first_valid_index()
    if first is None:
        return []
    start = series.index.get_loc(first)
    if not isinstance(start, int):
        return []
        
    return np.flip(np.arange(len(series) - 1, start, -spacing)).tolist()

T = TypeVar('T')
def iter_progress(input: Iterable[T], verbose: bool, desc: str | None) -> Iterable[T]:
    if verbose:
        import tqdm
        return tqdm.tqdm(input, desc=desc, leave=False)
    else:
        return input

class Dataset:
    
    def __init__(
        self: Self,
        name: str, 
        df: pd.DataFrame,
    ):
        self.name = name
        self.df = df
        df.index.name = "time"
    
    @staticmethod
    def load(
        name: str, 
        df: Callable[[], pd.DataFrame],
        age_col: Column,
        feature_cols: FeatureColumns,
        age_scale: int = 1,
    ):
        assert len(feature_cols) != 0
        
        feature_cols = __class__._convert_feature_cols(feature_cols)
        
        loaded_df = __class__._prepare_df(
            df(),
            age_col,
            age_scale,
            list(feature_cols.keys()),
        ).rename(feature_cols, axis=1)
        
        return Dataset(
            name=name,
            df=loaded_df,
        )
    
    def ages(self: Self) -> np.ndarray:
        return self.df.index.to_numpy(copy=False)
    
    def features(self: Self) -> pd.DataFrame:
        return self.df
    
    def feature_names(self: Self) -> pd.Index[str]:
        return self.df.columns
    
    def transform(self: Self, function: Callable[[pd.DataFrame], pd.DataFrame], name: str | None = None) -> Self:
        return self.__class__(
            name=name or self.name,
            df=function(self.df),
        )
        
    def age_range(self: Self, min: int | float | None = None, max: int | float | None = None, name: str | None = None) -> Self:
        return self.__class__(
            name=name or self.name,
            df=self.df[(self.df.index <= max) & (self.df.index >= min)],
        )
    
    def split(self: Self, features: dict[str, FeatureColumns]) -> dict[str, Self]:
        
        def map(name: str, columns: FeatureColumns) -> Self:
            columns = self._convert_feature_cols(columns)
            df = self.df[columns.keys()].rename(columns, axis=1)
            df.index.name = "time"
            return self.__class__(
                name=self.name + " " + name,
                df=df,
            )
        
        return {feat_name: map(feat_name, column) for feat_name, column in features.items()}
           
    # Clean up data frame before transforming
    @staticmethod
    def _prepare_df(
        df: pd.DataFrame,
        age_col: Column,
        age_scale: int | float,
        feature_cols: Sequence[Column],
    ) -> pd.DataFrame:
        
        # Select relevant columns and ensure age is present
        df_sel: pd.DataFrame = df[[age_col] + list(feature_cols)].dropna(subset=[age_col])
        df_sel[age_col] *= age_scale

        if df_sel.empty:
            raise ValueError(f"No rows remain after dropping rows with missing '{age_col}'. Check your data.")

        # If all feature values are NaN for every row, abort with helpful message
        if df_sel[feature_cols].dropna(how='all').empty:
            raise ValueError(
                "No feature values found for any of the requested feature columns. "
                "This may be caused by incorrect feature column names or a pivot that produced only NaNs. "
                f"Requested features: {feature_cols}"
            )

        df_sel[feature_cols] = df_sel[feature_cols].astype(float)

        # Final drop: remove rows that are still entirely NaN in the feature columns
        df_sel = df_sel.dropna(subset=feature_cols, how='all')

        if df_sel.empty:
            raise ValueError("No rows with usable feature values after imputation. Check your data and feature selection.")
        
        df_sel = df_sel.groupby(age_col, as_index=False).mean(numeric_only=True).set_index(age_col) # type: ignore
        df_sel.index.name = "time"

        return df_sel
    
    @staticmethod
    def age_format(ages: np.ndarray[tuple[Any], np.dtype[np.number]]) -> str:
        if ages.size == 0:
            raise ValueError("Empty timespan!")
        ages = np.abs(ages)
        ages = ages[np.nonzero(ages)]
        lowest = np.min(ages)
        match np.log10(lowest) // 3:
            case 0: return "ya"
            case 1: return "kya"
            case 2: return "mya"
            case _: return "?ya"
            
    @staticmethod
    def _convert_feature_cols(feature_cols: FeatureColumns) -> dict[Column, str]:
        return {col:col if isinstance(col, str) else " ".join(col) for col in feature_cols} if not isinstance(feature_cols, dict) else feature_cols

RESULTS = TypeVar('RESULTS')
class Model(Generic[RESULTS], metaclass = abc.ABCMeta):
    
    def __init__(self: Self, name: str):
        self.name = name
        self.results: dict[Dataset, RESULTS] = dict()
        
    @abc.abstractmethod
    def run(self: Self, dataset: Dataset):
        pass
    
    @abc.abstractmethod 
    def _plot(self: Self, dataset: Dataset) -> Figure:
        pass
    
    def run_with_output(self, datasets: Iterable[Dataset], path: str):
        from builtins import print as println
        
        println(f"###### Running {self.name} on datasets:")
        for dataset in datasets:
            try:
                println(f"### Running {dataset.name}...")
                start = time.time()
                self.run(dataset)
                println(f"### {self.name} completed in {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))}")
                println()
                
                try:
                    fig = self._plot(dataset)
                    import os.path
                    fig.savefig(os.path.join(path, f"{self.name} {dataset.name}.png"))
                    plt.close(fig)
                except Exception:
                    traceback.print_exc()
            except Exception:
                traceback.print_exc()
            
        println()
