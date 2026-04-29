import functools
from typing import Iterable, Literal, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats._axis_nan_policy import SmallSampleWarning
from . import Dataset, DetrendMethod, Model, compute_residuals, space_indices, time_index 
import warnings
        
class Metrics(Model[pd.DataFrame]):
    
    def __init__(
        self: Self, 
        span: float | int | None = None,
        window: float | int | None = None,
        detrend: DetrendMethod | None = None,
        ktau_sampling: int | Sequence[int] | None = None,
    ):
        super().__init__("Metric-based analysis")
        assert span is None or 0.0 < span
        
        self.span = span
        self.window = window
        self.detrend: DetrendMethod | None = detrend
        self.ktau_sampling = ktau_sampling

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
    
    def run_on_series(self: Self, series: pd.Series, span: int | float | None = None, window: float | int | None = None, detrend: DetrendMethod | None = None, ktau_sampling: int | Sequence[int] | None = None) -> pd.DataFrame:
        
        series = pd.Series(series.values, index=time_index(series.index))
        
        window = self.window_index(series, window or self.window or 0.25)
            
        residuals = compute_residuals(series, span=span or self.span or 0.2, method=detrend or self.detrend)
        rolling = residuals.rolling(window=window)
        variance = self.variance(rolling)
        ac1 = self.ac1(rolling)
    
        df = series.to_frame(name="value").join([residuals, variance, ac1])
        
        ktau_sampling = ktau_sampling or self.ktau_sampling
        if ktau_sampling is not None:
            for col in ["variance", "ac1"]:
                series = df[col]
                spacing = space_indices(series, ktau_sampling) if isinstance(ktau_sampling, int) else ktau_sampling
                df["ktau_" + col] = self.ktau(series, indices=spacing)
                
        df =  df.reset_index("time").set_index("time")    
        
        print(df)         
            
        return df
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
        self.results[dataset] = pd.concat((self.run_on_series(dataset.df[column]).reset_index() for column in dataset.df.columns)) # type: ignore               
    
    def _plot(self: Self, dataset: Dataset, title: bool = True, transitions: Iterable[int | float] = []) -> Figure:
        ROWS = 2
        fig, axes = pyplot.subplots(nrows=ROWS, sharex = True)
        __class__.plot(axes, self.results[dataset], transitions=transitions)
        if title:
            fig.suptitle("Metrics of " + dataset.name)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot(axs: Sequence[Axes], df: pd.DataFrame, transitions: Iterable[int | float] = [], age: str = "Age (ya BP)"):
        
        time = np.abs(time_index(df.index))
        
        axs[-1].set_xlabel(age)

        def plot(ax: Axes, data: pd.Series, title: str):
            ax.set_ylabel(title)
            ax.xaxis.set_inverted(True)
            ax.plot(time, data.to_numpy(), label=title)
            for transition in transitions:
                transition = np.abs(transition)
                if time.min() <= transition <= time.max():
                    ax.axvline(transition, linestyle='dashed')
                    
        plot(axs[0], df["variance"], "Variance")
        plot(axs[1], df["ac1"], "AC-1")
