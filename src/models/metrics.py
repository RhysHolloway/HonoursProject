import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from wrapper import Dataset, Model


class Metrics(Model):
    
    def __init__(
        self, 
        seq_len: int,
        variance: bool = True,
        autocorrelation: bool = True,
        ):
        super().__init__("Metric-based analysis")
        
        self.seq_len = seq_len
        self.variance = variance
        self.autocorrelation = autocorrelation
    
    def runner(
        self,
        datasets: list[Dataset],
    ):
        
        def single(dataset: Dataset):
            ages = dataset.ages()
            features = dataset.features()
            seq_features = pd.Series(np.array(features) / np.mean(np.abs(features)), index=ages[self.seq_len:])
            
                            
            def uniform():
                if not self.autocorrelation:
                    return False
                distance = ages[1] - ages[0]
                for i in range(2, len(ages)):
                    if ages[i] - ages[i - 1] != distance:
                        print(f"{dataset.name} is not uniform! Cannot compute autocorrelation.")
                        return False
                return True
            
            variance = seq_features.var() if self.variance else None
            autocorrelation = seq_features.autocorr(lag=1) if uniform() else None
        
            return variance, autocorrelation
        
        results = [(dataset, single(dataset)) for dataset in datasets]
        
        def print_single(result: tuple[Dataset, tuple]):
            dataset, result = result
            
            # def print(""):
            
            print(f"### Results for {dataset.name}")
                   
        
        def plot_single(result: tuple[Dataset, tuple]):
            dataset, (variance, autocorrelation) = result
            ages = dataset.ages()
            fig: Figure = matplotlib.pyplot.figure(figsize=(8, 4))
            fig.suptitle(dataset.name)
            axs: Axes = fig.subplots(sum([self.variance, self.autocorrelation] * 1), 1, sharex = True)
            
            i = 0
                
            if self.variance:
                axs[i].set_title("Variance")
                axs[i].plot(ages[self.seq_len:], variance)
                i+=1
            if self.autocorrelation:
                axs[i].set_title("Autocorrelation")
                axs[i].plot(dataset.ages(), autocorrelation)
                i+=1
                
            return fig
        
        return (lambda: map(print_single, results), lambda: list(map(plot_single, results)))