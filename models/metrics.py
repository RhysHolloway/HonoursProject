from typing import Literal, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats._axis_nan_policy import SmallSampleWarning
from models import Dataset, Model, compute_residuals, space_indices

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
    def ktau(series: pd.Series, spacing: int) -> pd.Series:        
        # Get first non-NaN index in series
        indices = space_indices(series, spacing)
        if len(indices) == 0:
            return pd.Series(name = f"ktau_{series.name}")
        start = indices.pop(0)
        
        def compute(imax: int) -> float:
            s = series.iloc[start:imax]
            import warnings
            with warnings.catch_warnings(action="ignore", category=SmallSampleWarning):
                return scipy.stats.kendalltau(s.index.to_numpy(), s.to_numpy())[0] # type: ignore
        # Return series
        return pd.Series(map(compute, indices), index=series.index[indices], name = f"ktau_{series.name}")
    
    @staticmethod
    def window_index(series: pd.Series, window: float) -> int:
        if window <= 0:
            raise ValueError("Rolling window is less than zero!")
        return int(window * len(series) if 0 < window <= 1 else window)
    
    @staticmethod
    def variance(rolling: pd.api.typing.Rolling) -> pd.Series:
        var: pd.Series = rolling.var()
        var.name = "variance"
        return var
        
    @staticmethod
    def std(rolling: pd.api.typing.Rolling) -> pd.Series:
        var: pd.Series = rolling.std()
        var.name = "std"
        return var
        
    @staticmethod
    def ac1(rolling: pd.api.typing.Rolling) -> pd.Series:
        auto: pd.Series = rolling.apply(func=lambda x: pd.Series(x).autocorr(lag=1), raw=True)
        auto.name = "ac1"
        return auto
    
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
                df["ktau_" + col] = self.ktau(df[col], spacing=ktau_distance)
            
        # df.insert(0, "variable", [series.name] * len(df))

        return df
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
        self.results[dataset] = pd.concat((self.run_on_series(dataset.df[column]).reset_index() for column in dataset.feature_cols.keys())) # type: ignore
        
    def _print(self: Self, dataset: Dataset):
        pass                   
    
    def _plot(self: Self, dataset: Dataset) -> Figure:
        results = self.results[dataset]
        ROWS = 5
        fig = pyplot.figure(figsize=(6 * len(dataset.feature_cols), 3 * ROWS))
        fig.suptitle(dataset.name)
        axs: Sequence[Axes] = fig.subplots(nrows=ROWS, ncols=1, sharex = True)
        
        def plot(data: pd.Series, title: str, i: int):
            axs[i].set_ylabel(title)
            axs[i].invert_xaxis()
            if i == 0:
                axs[i].legend(dataset.feature_names())
            if i == ROWS - 1:
                axs[i].set_xlabel(f"Age ({dataset.age_format})")
            if data is not None:
                # names = dataset.feature_names()
                # for j, data in enumerate(data):
                axs[i].plot(data, data.index, label=title)
                    
            plot(results["state"], "Data", 0)
            plot(results["variance"], "Variance", 1)
            plot(results["ac1"], "AC-1", 2)
            plot(results["ktau_variance"], "KTau Variance", 3)
            plot(results["ktau_ac1"], "KTau AC-1", 4)
            
        return fig