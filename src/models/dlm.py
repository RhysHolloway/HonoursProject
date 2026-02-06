from typing import Self
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pydlm import dlm, trend, dynamic, autoReg

from processing import Dataset, Model

class DLM(Model[dlm]):    
    
    def __init__(self, ar: int = 10, window: int = 50):
        super().__init__("DLM")    
        self.window = window

    def _run(
        self: Self,
        dataset: Dataset,
    ):
            
        features = dataset.features()
        data = pd.Series(features[:,0], index=dataset.ages())
        extra_features = dynamic(features=features[:,1:], name='extra_features') if len(features) > 1 else None 
    
        model = dlm(data, separatePlot=False) + trend(1) + autoReg(degree=10)
        
        model.options.separatePlot = False
        model.options.showConfidenceInterval = False
        
        if extra_features is not None:
            model = model + extra_features
            
        
        
        model.fitForwardFilter(useRollingWindow=True, windowLength=self.window)
        
        # return model
        
        return model
        
    def _print(self: Self, dataset: Dataset):
        pass

    def _plot(self: Self, dataset: Dataset) -> Figure:
        model = self.results[dataset]
        # fig = plt.figure(num=i, clear=True)
        model.plot()
        fig = plt.gcf()
        for ax in fig.axes:
            ax.set_title(f"{self.name} of {dataset.name}")
            ax.invert_xaxis()
            ax.set_xlabel(f"Age ({dataset.age_format()})")
        return fig