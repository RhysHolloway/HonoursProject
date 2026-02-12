from typing import Final, Self, Sequence, Callable
from models import Dataset, Model

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from keras.models import load_model, Sequential

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LSTMLoader:
    
    def __init__(self: Self, path: str, extension: str = ".keras"):
        import os
        import os.path
        import functools
        
        if not extension.startswith("."):
            extension = "." + extension
            
        def load(dir: str) -> list[Sequential]:
            dir = os.path.join(path, dir)
            if not os.path.exists(dir):
                return []
            return [load_model(os.path.join(dir, name)) for name in os.listdir(dir) if name.endswith(extension)]
        
        self._500models = load("len500")
        self._1500models = load("len1500")
        
        if sum(map(len, [self._500models, self._1500models])) == 0:
            raise ValueError(f"Could not load any {extension} models from {path}!")
        
        self.with_args = functools.partial(LSTM, get_models=self._get_models)
    
    def _get_models(self: Self, series: pd.Series):
        if len(self._1500models) == 0:
            return self._500models
        elif len(self._500models) == 0:
            return self._1500models
        else:
            return self._1500models if len(series) > 500 else self._500models

type LSTMResults = tuple[pd.DataFrame, pd.DataFrame, dict[str | None, dict[str, list[tuple]]]]
class LSTM(Model[LSTMResults]):
        
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

    def run(
        self: Self,
        dataset: Dataset,
    ):
               
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
        
        self.results[dataset] = (dl_preds, dl_preds_mean, dl_peaks)

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
    
    
from functools import reduce
from typing import Callable, Final, Iterable, Literal, Self, Tuple
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

type BifId = Literal["HB", "BP", "LP"]
type BifType = Tuple[BifId, bool]
BIFS: Final[list[BifId]] = ["HB", "BP", "LP"]

def bif_types():
    for null in [True, False]:
        for type in BIFS:
            yield (type, null)

def bif_maximum(type: BifType, bif_max: int) -> int:
    type, null = type
    if null:
        return ((bif_max - 2 * (bif_max // 3)) if (type == "BP") else (bif_max // 3))
    else:
        return bif_max

INDEX_COL: Final[str] = 'sequence_ID'
LABEL_COLS: Final[list[str]] = [INDEX_COL, 'class_label', 'bif', 'null']

type Sims = pd.Series
type Resids = pd.Series
type Labels = pd.DataFrame
type Groups = pd.DataFrame
type TrainData = tuple[dict[int, tuple[Sims, Resids]], Labels, Groups]

# Returns combined (indexed sims + resids, labels, groups)
def combine_batches(batches: Iterable[TrainData]) -> TrainData:
    
    def _reduce_combine(
        a: TrainData, 
        b: TrainData
    ) -> TrainData:
        seq_max = max(a[0].keys())
        new_labels = b[1].copy()
        new_labels.index += seq_max # index starts at 1 so no conflict between last key in previous and first key in next
        new_groups = b[2].copy()
        new_groups.index += seq_max
        return (a[0] | {key + seq_max:val for key, val in b[0].items()}, pd.concat([a[1], new_labels]), pd.concat([a[2], new_groups]))
    
    return reduce(_reduce_combine, batches)


def compute_residuals(data: pd.Series) -> pd.Series:
        smooth_data = lowess(data.values, data.index.values, frac=0.2)[:, 1]
        return pd.Series(data.values - smooth_data, index=data.index, name="Residuals")
    
type XY = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type DeFunc = Callable[[XY, float], XY]

class StochSim():

    def __init__(
        self: Self,
        binit: float,
        bcrit: float,
        ts_len: int,
        dt: np.float64 = 0.01,
        dt_sample: float = 1.0,
        t0: int = 0,
    ):
        """
        Stochastic simulator class
        
        :param self: Simulator class
        :type self: Self
        :param binit: Initial value of bifurcation parameter
        :type binit: float
        :param bcrit: Bifurcation point of model
        :type bcrit: float
        :param ts_len: Number of points in time series
        :type ts_len: int
        :param null_location: Null simulation (float, bifurcation parameter fixed) or
                transient simulation (None, bifurcation parameter increments to bifurcation point) 
                Float value in [0,1] to determine location along bifurcation branch
                where null is simulated. Value is proportion of distance to the 
                bifurcation from initial point (0 is initial point, 1 is bifurcation point)
        :type null_location: Union[float, None]
        :param dt: Description
        :type dt: np.float64
        :param dt_sample: Indices between sampled points
        :type dt_sample: int
        :param t0: Start time in time series
        :type t0: int
        """
    
        self.binit = binit
        self.bcrit = bcrit
        
        # Initialise arrays to store single time-series data
        assert t0 >= 0
        t = np.arange(t0, ts_len, dt)
        
        # Set up bifurcation parameter b, that increases linearly in time from binit to bcrit
        self.parameter = pd.Series(np.linspace(binit,bcrit,len(t)),index=t)
        
        self.dt = dt
        self.dt_sample = dt_sample
        

    # Throws floating point error
    def simulate(
        self: Self,
        de_fun: DeFunc,
        s0: XY,
        sigma: float | XY, 
        tburn: int = 100, # burn-in period
        clip: Callable[[XY], XY] = lambda s: s,
        rand: np.random.RandomState = np.random.mtrand._rand,
    ) -> pd.DataFrame:      
        """
        Function to run a stochastic simulation of model up to bifurcation point
        
        :param self: Initialized simulation class
        :type self: Self
        :param de_fun: Derivative equation to simulate
        :type de_fun: DeFunc
        :param s0: Initial value
        :type s0: XY
        :param sigma: amplitude factor of GWN - total amplitude also depends on parameter values
        :type sigma: float | XY
        :param tburn: Burn-in length
        :type tburn: int
        :param clip: Optional clipping function for simulation
        :type clip: Callable[[XY], XY]
        :return: DataFrame of trajectories indexed by time
        :rtype: DataFrame
        """
        
        ## Implement Euler Maryuyama for stocahstic simulation
        s: np.ndarray[np.ndarray[np.float64]] = np.zeros([len(self.parameter),2])
        
        # Create brownian increments (s.d. sqrt(dt))
        dW_burn = rand.normal(loc=0, scale=sigma*np.sqrt(self.dt), size = [int(tburn/self.dt),2])
        dW = rand.normal(loc=0, scale=sigma*np.sqrt(self.dt), size = [len(self.parameter/self.dt),2])
        
        # Run burn-in period on s0
        for i in range(int(tburn/self.dt)):
            s0 = s0 + de_fun(s0, self.parameter.iloc[0]) * self.dt + dW_burn[i]
            
        # Initial condition post burn-in period
        s[0] = s0
        
        # Run simulation
        for i in range(len(self.parameter)-1):
            # Update bifurcation parameter
            s[i+1] = clip(s[i] + de_fun(s[i], self.parameter.iloc[i])*self.dt + dW[i])
                
        # Store series data in a DataFrame
        df_traj = pd.DataFrame({"Time": self.parameter.index} | {f"p{i}":s[:,i] for i in range(s.shape[1])})#, 'b': self.parameter.to_numpy()})
        
        # Filter dataframe according to spacing
        df_traj_filt = df_traj.iloc[0::int(self.dt_sample/self.dt)]
        df_traj_filt["Time"] = df_traj_filt["Time"].astype(int)

        return df_traj_filt