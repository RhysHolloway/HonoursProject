import os
from typing import Self, Union, Callable, Any
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import find_peaks
from util import filter_points, get_project_path
from processing import Dataset, Model
from keras.models import load_model, Sequential
import ewstools

class LSTMModels:
    
    def __init__(self: Self, path: str, extension: str = ".h5"):
        root_path = get_project_path(path)
        self._500models = [load_model(root_path + "len500/" + name) for name in os.listdir(root_path + "len500/") if name.endswith(extension)]
        self._1500models = [load_model(root_path + "len1500/" + name) for name in os.listdir(root_path + "len1500/") if name.endswith(extension)]
    
    def get_models(self: Self, series: pd.Series):
        return self._1500models if len(series) > 500 else self._500models
    

# Make sure to do linear interpolation before using

class LSTM(Model):
        
    def __init__(
        self,
        get_models: Callable[[Any], list[Sequential]],
        thresholds: list[float],
        prominence: float,
        distance: int,
        ):
        super().__init__("CNN-LSTM")
        
        self.get_models = get_models
        self.thresholds = thresholds
        self.prominence = prominence
        self.distance = distance
        
    def __compute_ews(dataset: Dataset, dic_bandwith: dict) -> list[pd.DataFrame]:

        # # Bandwidth sizes for Gaussian kernel (used in Dakos (2008) Table S3)
        # dic_bandwidth = {
        #     "End of greenhouse Earth": 25,
        #     "End of Younger Dryas": 100,
        #     "End of glaciation I": 25,
        #     "Bolling-Allerod transition": 25,
        #     "End of glaciation II": 25,
        #     "End of glaciation III": 10,
        #     "End of glaciation IV": 50,
        #     "Desertification of N. Africa": 10,
        # }
            
        #     return (model, history)
        # EWS computation parameters
        rw = 0.5  # half the length of the data

        # Loop through each record

        list_df = []
        for feature in dataset.feature_cols:
            
            df = dataset.df

            # Make time negative so it increaes up to transition
            df[dataset.age_col] = -df[dataset.age_col]
            # Series for computing EWS
            series = df.set_index(dataset.age_col)[feature]

            ts = ewstools.TimeSeries(series)
            ts.detrend(method="Gaussian")#, bandwidth=)
            ts.compute_var(rolling_window=rw)
            ts.compute_auto(rolling_window=rw, lag=1)

            df_ews = ts.state.join(ts.ews)
            list_df.append(df_ews)

        # Concatenate dataframes
        df_ews = pd.concat(list_df)

        # # Export
        return df_ews
        
    def __classify(self, dataset: Dataset):
        
        df_ews_forced = self.__compute_ews(dataset)
        
        series = df_ews_forced["residuals"]
        
        ts = ewstools.TimeSeries(dataset.features())
        # space predictions apart by 10 data points (inc must be defined in terms of time)
        dt = series.index[1] - series.index[0]
        inc = dt * 10
        
        classifier_list = self.get_models(series)
        
        for i, classifier in enumerate(classifier_list):
            ts.apply_classifier_inc(classifier, inc=inc, verbose=1, name=str(i))

        # Get ensemble dl prediction
        dl_preds_mean = ts.dl_preds.groupby("time")[[0, 1, 2, 3]].mean()
        dl_preds_mean.columns = ["fold_prob", "hopf_prob", "branch_prob", "null_prob"]
        dl_preds_mean.index.name = "Time"
        
        def find_peaks_in_col(col, ages):
        
            height = np.percentile(col, min(self.thresholds))
            
            # Find and filter by prominence
            peaks, _ = find_peaks(
                col,
                height=height,
                prominence=self.prominence
            )
            
            # Filter by distance
            peaks = filter_points(
                points=peaks, 
                scores=col[peaks], 
                ages=ages, 
                min_distance=self.distance
            )
            
            return peaks
        
        points = [find_peaks_in_col(col, ts.dl_preds["Time"]) for col in ts.dl_preds[[0,1,2]]]
        
        return (
            ts.dl_preds, dl_preds_mean, points
        )
        
        
    TITLES = ["Fold", "Hopf", "Transcritical", "Mean Probability"]

    def runner(
        self,
        datasets: list[Dataset],
    ):        
        results = [(dataset, self.__classify(dataset)) for dataset in datasets]

        def print_single(tup: tuple):
            dataset, (preds, means, points) = tup

            for i, points in enumerate(points):
                print(f"Detected {len(points)} {self.TITLES[i]} points:")
                for point in points:
                    print(f"{preds[point][i]} at {preds[point]} {dataset.age_format}")
            
        def plot_single(tup: tuple):
            dataset, (preds, means, points) = tup
                    
            fig = plt.figure(4, 12, sharex = True)
            fig.suptitle(f"Bifurcation classifications on {dataset.name}")
            subplots = fig.subplots(nrows=4, ncols=1, sharex=True)
            for i, ax in enumerate(subplots):
                ax.invert_xaxis()
                ax.set_title(self.TITLES[i])
                ax.set_xlabel(f"Age ({dataset.age_format})")
                ax.set_ylabel("Probability")
                ax.plot(preds[i] if i != 3 else means, preds["Time"])
                if i != 3:
                    ax.scatter(points, preds["Time"])
                
                for threshold in self.thresholds:
                    ax.axhline(threshold, linestyle="--")

            return fig
            
        return (lambda: map(print_single, results), lambda: list(map(plot_single, results)))

    