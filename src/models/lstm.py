from typing import Final, Self, Sequence, Callable
from util import get_project_path
from processing import Dataset, Model

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

class LSTMLoader:
    
    def __init__(self: Self, path: str, extension: str = "keras"):
        import os
        import os.path
        import functools
        extension = "." + extension
        root_path = get_project_path(path)
        
        def load(dir: str) -> list[Sequential]:
            dir = os.path.join(root_path, dir)
            return [load_model(os.path.join(dir, name)) for name in os.listdir(dir) if name.endswith(extension)]
        
        self._500models = load("len500")
        self._1500models = load("len1500")
        
        self.with_args = functools.partial(LSTM, get_models=self._get_models)
    
    def _get_models(self: Self, series: pd.Series):
        return self._1500models if len(series) > 500 else self._500models
        # print("Warning: loading only 1500 length models")
        # return self._1500models

type _Results = tuple[pd.DataFrame, pd.DataFrame, dict[str | None, dict[str, list[tuple]]]]
class LSTM(Model[_Results]):
        
    def __init__(
        self: Self,
        get_models: Callable[[pd.Series], list[Sequential]],
        bandwidth: float = 0.2,
        prominence: float = 0.05,
        distance: int = 10,
        verbose: bool = True,
    ):
        super().__init__("CNN-LSTM")
        
        self.get_models = get_models
        self.bandwidth = bandwidth
        self.prominence = prominence
        self.distance = distance
        self.verbose = verbose
        
    COLUMNS: Final[list[str]] = ["Fold", "Hopf", "Branch", "Null"]

    def _run(
        self: Self,
        dataset: Dataset,
    ) -> _Results:
               
        ages = -dataset.ages()
        list_df: list[pd.DataFrame] = []
        for col, feature in dataset.features().items():
            series = pd.Series(feature.to_numpy(), index=ages)[::-1]
                             
            classifier_list = self.get_models(series)        
            if len(classifier_list) == 0:
                raise ValueError(f"No LSTM models were found to be used with {dataset.name}!")
            
            # Calculate residual (from ewstools)
            # Standard deviation of kernel given bandwidth
            # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
            bandwidth = self.bandwidth * len(series) if 0 < self.bandwidth <= 1 else self.bandwidth
            series -= gaussian_filter(series.to_numpy(), sigma=(0.25 / 0.675) * bandwidth, mode="reflect")

            for i, classifier in enumerate(classifier_list):
                            
                # space predictions apart by 10 data points (inc must be defined in terms of time)
                dt = series.index[1] - series.index[0]
                inc = dt * 10
                tmin = series.index[0]
                tend = series.index[-1] # if not self.transition else self.transition
                tend += dt  # Make transition point inclusive

                model_name = f"len{classifier.input_shape[1]}-{i}"
                
                tmax_dfs: list[pd.DataFrame] = []
                for tmax in np.arange(tmin + inc, tend + dt, inc):
                    
                    input_series: pd.Series = series[(series.index >= tmin) & (series.index < tmax)]
                    input_series = input_series / (abs(input_series).mean())
                    
                    # BURY PAPER RESTRICTS TIME SERIES TO LAST 500/1500 STEPS BEFORE TRANSITION
                    input_data: np.ndarray = (
                        input_series.iloc[-classifier.input_shape[1]:].to_numpy() \
                        if len(input_series) > classifier.input_shape[1] else \
                            np.concatenate(
                                (np.zeros(classifier.input_shape[1] - len(input_series)), input_series.to_numpy())
                            )
                        ).reshape(1, -1, 1)

                    # Get DL prediction
                    dl_pred: np.ndarray = classifier.predict(input_data, verbose=self.verbose)[0]
                    # Put info into dataframe
                    df = pd.DataFrame({self.COLUMNS[i]:v for i, v in enumerate(dl_pred)}, index = [0])
                    df.insert(0, "Time", -input_series.last_valid_index())
                    df["tmax"] = tmax
                    tmax_dfs.append(df)
                df = pd.concat(tmax_dfs)
                df.insert(0, "Feature", dataset.feature_name(col))
                df.insert(1, "Model", model_name)
                df.insert(len(df.columns) - 1, "tmin", tmin)
                list_df.append(df)
                    
        if len(list_df) == 0:
            raise ValueError("Could not compute any predictions for", dataset.name)

        dl_preds: pd.DataFrame = pd.concat(list_df)

        # Get ensemble dl prediction (if multiple features)
        dl_preds_mean = dl_preds.groupby("Time")[self.COLUMNS].mean()
        dl_preds_mean.index.name = "Time"
        
        def peaks(df: pd.DataFrame) -> dict[str, list[tuple]]:
            
            def find_peaks_in_col(col: pd.Series):

                # Find and filter by prominence
                ages = col.index
                height = col.to_numpy()
                peak_indices, _ = find_peaks(
                    ages,
                    height=height,
                    prominence=self.prominence,
                    distance=self.distance
                )
                
                return list(zip(ages[peak_indices], height[peak_indices]))
            
            return {
                name:find_peaks_in_col(col)
                for name, col in df[self.COLUMNS[:-1]].items()
            }
            
        dl_peaks = {feature:peaks(dl_preds[dl_preds["Feature"] == feature].set_index("Time")) for feature in dataset.feature_names()} | {None:peaks(dl_preds_mean)}
        
        return (dl_preds, dl_preds_mean, dl_peaks)

    def _print(self: Self, dataset: Dataset):
        _, _, points = self.results[dataset]
        for feature, points in points.items():
            detected = sum(len(p) for p in points.values())
            if detected != 0:
                print(f"Detected {detected} peaks for {"feature " + feature if feature is not None else "combined data"}:")
                for type, points in points.items():
                    for point in points:
                        print(f"{point[1]} ({type}) at {point[0]} {dataset.age_format}")
        
    def _plot(self: Self, dataset: Dataset) -> Figure:
                
        fig = plt.figure(figsize=(9, 12))
        fig.suptitle(f"Bifurcation classifications on {dataset.name}")
        
        preds, means, peaks = self.results[dataset]
        
        def plot_peak(ax: Axes, feature: None | str, col: str):
            points = peaks[feature][col]
            if len(points) != 0: 
                x, y = zip(*points)
                ax.scatter(x, y)
        
        subplots: Sequence[Axes] = fig.subplots(nrows=len(self.COLUMNS), ncols=1, sharex='all',)
        
        mean_ax = subplots[-1]
        mean_ax.invert_xaxis()
        mean_ax.set_xlabel(f"Age ({dataset.age_format})")
        mean_ax.set_ylabel("Mean Feature Probability")
        
        for i, col in enumerate(self.COLUMNS[:-1]): # All besides null column, which
            ax = subplots[i]
            ax.set_ylabel(col + " Probability")
            for feature in dataset.feature_names():
                feature_preds = preds[preds["Feature"] == feature].set_index("Time")[col]
                ax.plot(feature_preds)
                plot_peak(ax, feature, col)
            
            mean_ax.plot(means[col], label=col)
            plot_peak(ax, None, col)
            
        fig.legend(dataset.feature_names())
        
        mean_ax.legend()

        return fig

    