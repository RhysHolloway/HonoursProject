from typing import Any, Optional, Self, Sequence
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from models import T, Dataset, Model
from metrics import Metrics, MetricsResults
from lstm import LSTM, LSTMResults

type AggregateResults = None
class Aggregate(Model[AggregateResults]):
    
    def __init__(
        self,
        models: Sequence[Model],
    ):
        super().__init__("Aggregate analysis")
        self.models = models
    
    def run(
        self: Self,
        dataset: Dataset,
    ):

        
        def get(model: Model[T]) -> tuple[T, str]:
            if dataset not in model.results:
                model.run(dataset)
            return (model.results[dataset], model.name)
        
        for result, name in map(get, self.models):
            match name:
                case LSTM.name:
                    result: LSTMResults
                    
                case Metrics.name:
                    result: MetricsResults
                case _:
                    pass
             
        
    def _print(self, dataset: Dataset):
        pass
    
    def _plot(self: Self, dataset: Dataset) -> Figure:
        return pyplot.figure()