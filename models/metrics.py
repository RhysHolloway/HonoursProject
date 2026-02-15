from typing import Any, Callable, Literal, Optional, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import scipy.stats
from models import Dataset, Model

import ewstools

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
    def ktau(series: pd.Series, indices: Callable[[pd.Series], pd.Index] | pd.Index = lambda series: series.index, name: str | None = None) -> pd.Series:
        indices = indices(series) if callable(indices) else indices
        
        # Get first non-NaN index in series
        start = series[pd.notna(series)].index[1]
        
        def compute(t_end) -> float:
            s = series.loc[start:t_end]
            import warnings
            with warnings.catch_warnings(action="ignore", category=scipy.stats._axis_nan_policy.SmallSampleWarning):
                return scipy.stats.kendalltau(s.index.to_numpy(), s.to_numpy())[0]
        
        # Return series
        return pd.Series(map(compute, indices), index=indices, name=name)
    
    def run_on_series(self: Self, series: pd.Series, transition = None) -> pd.DataFrame:
        
        ts = ewstools.TimeSeries(series, transition=transition or self.transition)
        ts.detrend(method=self.detrend_method)
        
        ts.compute_var(rolling_window=self.window)
        ts.compute_auto(lag=1, rolling_window=self.window)
        
        STATE_COLS = ["state", "smoothing", "residuals"]
        EWS_COLS = ["variance", "ac1"]
        
        df = pd.concat([ts.state[STATE_COLS], ts.ews[EWS_COLS]], axis=1).rename({"state":"value"})
        
        if self.ktau_distance:
            indices = df.index[::self.ktau_distance]
            for col in EWS_COLS:
                df["ktau_" + col] = self.ktau(df[col], indices)
            
        df.insert(0, "variable", series.name)

        return df
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
                        
        # def uniform(ages):
        #     if not self.ac1:
        #         return False
        #     distance = ages[1] - ages[0]
        #     for i in range(2, len(ages)):
        #         if ages[i] - ages[i - 1] != distance:
        #             print(f"{dataset.name} is not uniform! Cannot compute autocorrelation.")
        #             return False
        #     return True
        
        # unif = uniform(dataset.ages())
            
        self.results[dataset] = pd.concat((self.run_on_series(dataset.df[column]).reset_index() for column in dataset.feature_cols.keys()))
        
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
                names = dataset.feature_names()
                for j, data in enumerate(data):
                    axs[i].plot(data, data.index, label=names[j])
                    
            plot(results["state"], "Data", 0)
            plot(results["variance"], "Variance", 1)
            plot(results["ac1"], "AC-1", 2)
            plot(results["ktau_variance"], "KTau Variance", 3)
            plot(results["ktau_ac1"], "KTau AC-1", 4)
            
        return fig