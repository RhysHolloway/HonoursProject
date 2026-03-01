import traceback
from typing import Callable, Generic, Iterable, Self, Sequence, TypeVar
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import time
import abc

type Column = str | Sequence[str]
type FeatureColumns = dict[Column, str] | Sequence[Column]

def resample_df(df: pd.DataFrame, feature_cols: Sequence[Column], steps: float = 1.0) -> pd.DataFrame:
    ages = df.index.to_numpy(dtype=float)
    new_ages = np.arange(np.ceil(ages.min()), np.floor(ages.max()), steps)
    return pd.DataFrame({col:np.interp(new_ages, ages, df[col].to_numpy()) for col in feature_cols}, index=new_ages)

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
        feature_cols: FeatureColumns,
        age_scale: int = 1,
    ):
        assert len(feature_cols) != 0
        
        feature_cols = __class__._convert_feature_cols(feature_cols)
        
        loaded_df = __class__._prepare_df(df(), age_col, feature_cols.keys())   
        
        return Dataset(
            name=name,
            age_format=__class__._get_age_format(loaded_df.index, age_scale),
            df=loaded_df,
            feature_cols=feature_cols,
        )
    
    def ages(self: Self) -> np.ndarray:
        return self.df.index.to_numpy(copy=False)
    
    def features(self: Self) -> pd.DataFrame:
        return self.df[self.feature_cols.keys()]
    
    def feature_names(self: Self) -> Iterable[str]:
        return self.feature_cols.values()
    
    def feature_name(self: Self, col: Column) -> str:
        return self.feature_cols[col]
    
    def transform(self: Self, function: Callable[[pd.DataFrame, Iterable[Column]], pd.DataFrame]) -> Self:
        return self.__class__(
                name=self.name,
                df=function(self.df, self.feature_cols.keys()),
                feature_cols=self.feature_cols,
                age_format=self.age_format
        )
    
    def split(self: Self, features: dict[str, FeatureColumns]) -> list[Self]:
        
        def map(name: str, columns: FeatureColumns) -> Self:
            columns = self._convert_feature_cols(columns)
            return self.__class__(
                name=self.name + " " + name,
                df=self.df[columns.keys()],
                feature_cols=columns,
                age_format=self.age_format
            )
        
        return [map(feat_name, column) for feat_name, column in features.items()]
           
    # Clean up data frame before transforming
    @staticmethod
    def _prepare_df(
        df: pd.DataFrame,
        age_col: Column,
        feature_cols: Column,
    ) -> pd.DataFrame:
        
        # Select relevant columns and ensure age is present
        df_sel: pd.DataFrame = df[[age_col] + list(feature_cols)]
        df_sel = df_sel.sort_values(by=age_col).reset_index(drop=True).dropna(subset=[age_col])

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

        return df_sel.set_index(age_col)
    
    @staticmethod
    def _get_age_format(ages: Sequence[int | float], age_scale: int):
        match np.log10(min(age for age in ages if age > 0) * age_scale) // 3:
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
    def _print(self: Self, dataset: Dataset):
        pass
       
    @abc.abstractmethod 
    def _plot(self: Self, dataset: Dataset) -> Figure:
        pass
    
    def run_with_output(self, datasets: Iterable[Dataset], print: bool = True, plot: bool = True, plot_path: str | None = None):
        from builtins import print as println
        
        println(f"###### Running {self.name} on datasets:")
        for dataset in datasets:
            try:
                println(f"### Running {dataset.name}...")
                start = time.time()
                self.run(dataset)
                println(f"### {self.name} completed in {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))}")
                println()
        
                if print:
                    println(f"###### {self.name} results for {dataset.name}:")
                    try:
                        self._print(dataset)            
                    except Exception:
                        println(f"###### {self.name} results for {dataset.name} failed with exception:")
                        traceback.print_exc()
                    println()
                
                if plot:
                    try:
                        fig = self._plot(dataset)
                        if plot_path is not None:
                            import os.path
                            fig.savefig(os.path.join(plot_path, f"{self.name} {dataset.name}.png"))           
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()
            
        println()