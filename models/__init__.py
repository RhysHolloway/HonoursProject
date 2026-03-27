import traceback
from typing import Any, Callable, Generic, Iterable, Literal, Self, Sequence, TypeVar, ValuesView
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import time
import abc

type Column = str | Sequence[str]
type FeatureColumns = dict[Column, str] | Sequence[Column]

def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    step = np.median([y-x for x, y in zip(df.index[:-1], df.index[1:])]) # Get the median distance between consecutive ages
    ages = df.index.to_numpy(dtype=float)
    new_ages = np.arange(np.ceil(ages.min()), np.floor(ages.max()), step)
    return pd.DataFrame({col:np.interp(new_ages, ages, series) for col, series in df.items()}, index=pd.Index(new_ages, name=df.index.name))

# def interpolate(df: pd.DataFrame, tcrit: float | None) -> pd.DataFrame:
#     """
#     Get data prior to the transition
#     Do linear interpolation to make data equally spaced

#     Input:
#         df: DataFrame with cols ['Age','Proxy','Transition']
#     Output:
#         df_inter: DataFrame of interpolated data prior to transition.
#             Has cols ['Age','Proxy','Transition']
#     """

#     # Get points prior to transition
#     df_prior: pd.DataFrame = df[df.index >= tcrit].copy()

#     # Equally spaced time values with same number of points as original record
#     t_inter_vals = np.linspace(
#         df_prior.index[0], df_prior.index[-1], len(df_prior)
#     )
#     # Make dataframe for interpolated data
#     df_inter = pd.DataFrame({df.index.name: t_inter_vals, "Inter": True}).set_index(df.index.name)
#     # Concatenate with original, and interpolate
#     df2 = pd.concat([df_prior, df_inter])
#     df2 = df2.interpolate(method="index")

#     # Extract just the interpolated data
#     df_inter = df2[df2["Inter"] == True][df.columns]

#     return df_inter

def compute_residuals(data: pd.Series, span: float = 0.2, type: Literal["Gaussian", "Lowess"] = "Lowess") -> pd.Series:
    from scipy.ndimage import gaussian_filter
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothing: np.ndarray
    match type:
        case "Gaussian":
            # Calculate residual (from ewstools)
            # Standard deviation of kernel given bandwidth
            # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
            span = span * len(data) if 0 < span <= 1 else span
            smoothing = gaussian_filter(data, sigma=(0.25 / 0.675) * span, mode="reflect")
        case "Lowess":
            span = span if 0 < span <= 1 else span / len(data)
            smoothing = lowess(data.to_numpy(), data.index.to_numpy(), frac=span, return_sorted=False)
    return pd.Series(data.to_numpy() - smoothing, index=data.index, name="residuals")

def space_indices(series: pd.Series, spacing: int) -> list[int]:
    start = series.index.get_loc(series.first_valid_index())
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
        feature_cols: dict[Column, str],
        age_format: str,
    ):
        self.name = name
        self.df = df
        self.feature_cols = feature_cols
        self.age_format = age_format
        
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
        
        loaded_df = __class__._prepare_df(df(), age_col, list(feature_cols.keys()))   
        
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
    
    def feature_names(self: Self) -> ValuesView[str]:
        return self.feature_cols.values()
    
    def feature_name(self: Self, col: Column) -> str:
        return self.feature_cols[col]
    
    def transform(self: Self, function: Callable[[pd.DataFrame], pd.DataFrame]) -> Self:
        return self.__class__(
                name=self.name,
                df=function(self.df),
                feature_cols=self.feature_cols,
                age_format=self.age_format
        )
    
    def split(self: Self, features: dict[str, FeatureColumns]) -> dict[str, Self]:
        
        def map(name: str, columns: FeatureColumns) -> Self:
            columns = self._convert_feature_cols(columns)
            df = self.df[columns.keys()]
            df.index.name = "time"
            return self.__class__(
                name=self.name + " " + name,
                df=df,
                feature_cols=columns,
                age_format=self.age_format
            )
        
        return {feat_name: map(feat_name, column) for feat_name, column in features.items()}
           
    # Clean up data frame before transforming
    @staticmethod
    def _prepare_df(
        df: pd.DataFrame,
        age_col: Column,
        feature_cols: Sequence[Column],
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

        df_sel[feature_cols] = df_sel[feature_cols].astype(float)

        # Final drop: remove rows that are still entirely NaN in the feature columns
        df_sel = df_sel.dropna(subset=feature_cols, how='all')

        if df_sel.empty:
            raise ValueError("No rows with usable feature values after imputation. Check your data and feature selection.")
        
        df_sel = df_sel.set_index(age_col)
        df_sel.index.name = "time"

        return df_sel
    
    @staticmethod
    def _get_age_format(ages: Iterable[Any], age_scale: int):
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