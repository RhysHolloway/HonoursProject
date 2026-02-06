from typing import Any, Callable, Generic, Self, Sequence, TypeVar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import time
from util import get_project_path, join_path
import abc

OUTPUT_PATH = get_project_path("output/plots/")
PRINT, PLOT = True, True

type _Col = str | Sequence[str]

class Dataset:
    
    def __init__(
        self, 
        name: str, 
        df: Callable[[], pd.DataFrame],
        age_col: _Col,
        feature_cols: dict[_Col, str | None] | Sequence[_Col],
        transform: Callable[[pd.DataFrame, _Col, _Col], pd.DataFrame] = lambda df, age_col, feature_cols : df, # Transform the input data
        age_scale: int = 1,
    ):
        assert len(feature_cols) != 0
        
        self.name = name
        self._age_col = age_col
        
        if isinstance(feature_cols, dict):
            self._feature_cols = list(feature_cols.keys())
            self._feature_names = [" ".join(list(col)) if name is None else name for col, name in feature_cols.items()]
        else:
            self._feature_cols = feature_cols
            self._feature_names = [" ".join(list(col)) for col in feature_cols]
        
        # Clean up data frame before transforming
        def prepare_df(
            df: pd.DataFrame,
            age_col: _Col,
            feature_cols: _Col,
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
        self.df = transform(self.df, self._age_col, self._feature_cols)
        self.df = prepare_df(self.df, self._age_col, self._feature_cols)
            
        match np.log10(min(age for age in self.ages() if age > 0) * age_scale) // 3:
            case 0: self._age_format = "ya"
            case 1: self._age_format = "kya"
            case 2: self._age_format = "mya"
            case _: self._age_format = "?ya"
        
    
    def ages(self: Self) -> np.ndarray:
        return self.df[self._age_col].to_numpy(copy=False)
        
    def features(self: Self) -> np.ndarray:
        return self.df[self._feature_cols].to_numpy(copy=False)
    
    def feature_names(self: Self) -> Sequence[str]:
        return self._feature_names
    
    def age_format(self: Self) -> str:
        return self._age_format

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
            print(f"### Running {dataset.name}...")
            start = time.time()
            self.results[dataset] = self._run(dataset)
            print(f"### {self.name} completed in {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))}")
            print()
        
        if PRINT:
            for dataset in datasets:
                print(f"###### {self.name} results for {dataset.name}:")
                self._print(dataset)
                print()
        
        if PLOT:
            for dataset in datasets:
                fig = self._plot(dataset)
                fig.savefig(join_path(OUTPUT_PATH, f"{self.name} {dataset.name.split(" ")[0]}.png"))
            
        print()