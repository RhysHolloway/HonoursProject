from typing import Any, Optional, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from models import Dataset, Model

import ewstools

type _Metric = Optional[pd.Series]

type MetricsResults = tuple[Optional[list[pd.Series]]]
class Metrics(Model[MetricsResults]):
    
    def __init__(
        self, 
        window: int,
        variance: bool = True,
        ac1: bool = True,
        ):
        super().__init__("Metric-based analysis")
        
        self.window = window
        self.variance = variance
        self.ac1 = ac1
        self.count = sum((self.variance, self.ac1))
        
        if self.count == 0:
            raise ValueError("Please provide ")
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
                        
        def uniform(ages):
            if not self.ac1:
                return False
            distance = ages[1] - ages[0]
            for i in range(2, len(ages)):
                if ages[i] - ages[i - 1] != distance:
                    print(f"{dataset.name} is not uniform! Cannot compute autocorrelation.")
                    return False
            return True
        
        unif = uniform(dataset.ages())
        
        def per_column(column: np.ndarray) -> tuple[_Metric, _Metric]:
                    
            features = pd.Series(np.array(column) / np.mean(np.abs(column)), index= -dataset.ages())[::-1]
            
            features = ewstools.TimeSeries(features)
            features.detrend()
            
            if self.variance:
                features.compute_var(rolling_window=self.window)
                variance = features.ews['variance']
            else:
                variance = None
            if self.ac1 and unif:
                features.compute_auto(lag=1, rolling_window=self.window)
                ac1 = features.ews['ac1']
            else:
                ac1 = None
    
            return variance, ac1
        
        variance, ac1 = zip(*(per_column(column) for column in dataset.features().to_numpy(copy=False).T))
        
        update = lambda metric: None if any(series is None for series in metric) else list(metric)
        
        variance = update(variance)
        ac1 = update(ac1)
        
        self.results[dataset] = (variance, ac1)
        
    def _print(self: Self, dataset: Dataset):
        pass                   
    
    def _plot(self: Self, dataset: Dataset) -> Figure:
        results = self.results[dataset]
        rows = sum(metric is not None for metric in results)
        fig = pyplot.figure(figsize=(8, 3 * rows))
        fig.suptitle(dataset.name)
        axs: Sequence[Axes] = fig.subplots(nrows=rows, ncols=1, sharex = True)
        
        variance, ac1 = results
        
        i = 0
        
        def plot(data: Optional[Any], title: str):
            axs[i].set_ylabel(title)
            axs[i].invert_xaxis()
            if i == 0:
                axs[i].legend(dataset.feature_names())
            if i == rows - 1:
                axs[i].set_xlabel(f"Age ({dataset.age_format})")
            if data is not None:
                names = dataset.feature_names()
                for j, data in enumerate(data):
                    axs[i].plot(data, label=names[j])
                    
            
        if self.variance:
            plot(variance, "Variance")
            i+=1
        if self.ac1:
            axs[i].invert_yaxis()
            plot(ac1, "AC-1 (flipped)")
            i+=1
            
        return fig