import functools
from typing import Literal, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats._axis_nan_policy import SmallSampleWarning
from models import Dataset, Model, compute_residuals, space_indices
import warnings
        
class Metrics(Model[pd.DataFrame]):
    
    def __init__(
        self, 
        window: float,
        detrend_method: Literal["Gaussian", "Lowess"] = "Lowess",
        ktau_distance: int | None = None,
        transition = None,
    ):
        super().__init__("Metric-based analysis")
        assert 0.0 < window < 1.0
        
        self.window = window
        self.detrend_method = detrend_method
        self.ktau_distance = ktau_distance
        self.transition = transition

    # Compute kendall tau values at equally spaced time points
    @staticmethod
    def ktau(series: pd.Series, indices: Sequence[int]) -> pd.Series:
        start = series.index.get_loc(series.first_valid_index())    
        if len(indices) == 0 or start is None:
            return pd.Series(dtype=float, name=f"ktau_{series.name}")
        new_series = functools.partial(pd.Series, dtype=float, index=series.index[indices], name=f"ktau_{series.name}")
        
        def compute(imax: int) -> float:
            if imax < start: # type: ignore
                return np.nan
            part = series.iloc[start:imax + 1]
            with warnings.catch_warnings(action="ignore", category=SmallSampleWarning):
                return scipy.stats.kendalltau(x=part.index, y=part).statistic # type: ignore
            
        return new_series(map(compute, indices)).dropna()
    
    @staticmethod
    def window_index(series: pd.Series, window: float) -> int:
        if window <= 0:
            raise ValueError("Rolling window is less than zero!")
        return int(window * len(series) if 0 < window <= 1 else window)
    
    @staticmethod
    def variance(rolling: pd.api.typing.Rolling) -> pd.Series:
        return pd.Series(rolling.var(), name = "variance")
        
    @staticmethod
    def std(rolling: pd.api.typing.Rolling) -> pd.Series:
        return pd.Series(rolling.std(), name = "std")
        
    @staticmethod
    def ac1(rolling: pd.api.typing.Rolling) -> pd.Series:
        return pd.Series(rolling.apply(func=lambda x: pd.Series(x).autocorr(lag=1), raw=True), name = "ac1")
    
    def run_on_series(self: Self, series: pd.Series, transition: int | None = None, ktau_distance: int | None = None, window: float | None = None) -> pd.DataFrame:
        
        series = series.copy(deep=False)
        series.index = series.index.get_level_values("time")
        
        if transition is not None:
            series = series[series.index <= transition]
        
        if window is None:
            window = self.window
        window = self.window_index(series, window)
            
        residuals = compute_residuals(series, type=self.detrend_method) # type: ignore
        rolling = residuals.rolling(window=window)
        variance = self.variance(rolling)
        ac1 = self.ac1(rolling)
    
        df = series.to_frame(name="value").join([residuals, variance, ac1])
        
        if ktau_distance is None:
            ktau_distance = self.ktau_distance
        
        if ktau_distance:
            for col in ["variance", "ac1"]:
                series = df[col]
                spacing = space_indices(series, ktau_distance)
                df["ktau_" + col] = self.ktau(series, indices=spacing)
            
        # df.insert(0, "variable", [series.name] * len(df))

        return df
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
        self.results[dataset] = pd.concat((self.run_on_series(dataset.df[column]).reset_index() for column in dataset.feature_cols.keys())) # type: ignore               
    
    def _plot(self: Self, dataset: Dataset, title: bool = True) -> Figure:
        ROWS = 2
        fig, axes = pyplot.subplots(nrows=ROWS, sharex = True)
        __class__.plot(axes, self.results[dataset])
        if title:
            fig.suptitle("Metrics of " + dataset.name)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot(axs: Sequence[Axes], df: pd.DataFrame):
        time = df.index.get_level_values("time")
        axs[-1].set_xlabel(f"Age (ya BP)")

        def plot(data: pd.Series, title: str, i: int):
            axs[i].set_ylabel(title)
            axs[i].invert_xaxis()
            axs[i].plot(time, data.to_numpy(), label=title)
                    
        plot(df["variance"], "Variance", 0)
        plot(df["ac1"], "AC-1", 1)
