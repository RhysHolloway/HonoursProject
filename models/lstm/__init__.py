from typing import Final, Self, Sequence, Callable
from models import Column, Dataset, Model

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras.models import load_model, Model as KerasModel

class LSTMLoader:
    
    def __init__(self: Self, path: str):
        import os
        import os.path
        import functools
        
        EXTENSIONS = [".keras", ".h5"]
            
        def load(dir: str) -> list[KerasModel]:
            dir = os.path.join(path, dir)
            if not os.path.exists(dir):
                return []
            return [load_model(os.path.join(dir, name)) for name in os.listdir(dir) if any(name.endswith(ext) for ext in EXTENSIONS)]
        
        self._500models = load("len500")
        self._1500models = load("len1500")
        
        if sum(map(len, [self._500models, self._1500models])) == 0:
            raise ValueError(f"Could not load any models from {path}!")
        
        self.with_args = functools.partial(LSTM, get_models=self._get_models)
    
    def _get_models(self: Self, series: pd.Series):
        if len(self._1500models) == 0:
            return self._500models
        elif len(self._500models) == 0:
            return self._1500models
        else:
            return self._1500models if len(series) > 500 else self._500models

type LSTMPeaks = dict[Column, dict[str, Sequence[int]]]
class LSTM(Model[pd.DataFrame]):
        
    def __init__(
        self: Self,
        get_models: Callable[[pd.Series], Sequence[KerasModel]],
        bandwidth: float = 0.2,
        verbose: bool = True,
    ):
        super().__init__("CNN-LSTM")
        
        self.get_models = get_models
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.result_means: dict[Dataset, LSTMPeaks] = dict()
        self.result_peaks: dict[Dataset, LSTMPeaks] = dict()
    
    COLUMNS: Final[list[str]] = ["Fold", "Hopf", "Branch", "Null"]
    
    
    def run_on_series(
        self: Self,
        series: pd.Series,
        spacing: int = 10
    ):
        list_df: list[pd.DataFrame] = []
        series = pd.Series(series.to_numpy(), index=-series.index, name=series.name)[::-1]
                             
        model_list = self.get_models(series)
        if len(model_list) == 0:
            raise ValueError(f"No LSTM models were found to be used with {series.name}!")
        
        # Calculate residual (from ewstools)
        # Standard deviation of kernel given bandwidth
        # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
        bandwidth = self.bandwidth * len(series) if 0 < self.bandwidth <= 1 else self.bandwidth
        series -= gaussian_filter(series.to_numpy(), sigma=(0.25 / 0.675) * bandwidth, mode="reflect")

        for model in model_list:

            tmax_dfs: list[pd.DataFrame] = []
            for tmax in np.arange(spacing, len(series), spacing):
                
                input = series[series.index[:tmax]].to_numpy()
                input = input / np.mean(np.abs(input))
                
                # Set time series to input shape either by truncation of the start of the series or by appending zeroes to the start
                if len(input) > model.input_shape[1]:
                    input = input[-model.input_shape[1]:]
                else:
                    input = np.concatenate((np.zeros(model.input_shape[1] - len(input)), input))

                # Get DL prediction
                df = pd.DataFrame(model.predict(input.reshape(1, model.input_shape[1], 1), verbose=self.verbose), columns=self.COLUMNS, index = [0])
                df.insert(0, "time", -series.index[tmax])
                tmax_dfs.append(df)
                
            df = pd.concat(tmax_dfs)
            list_df.append(df)
            
        df = self.calc_means(pd.concat(list_df))
        df.insert(0, "variable", series.name)
        return df
    
    @staticmethod
    def calc_means(predictions: pd.DataFrame) -> pd.DataFrame:
        return predictions.groupby("time")[__class__.COLUMNS].mean()
    
    @staticmethod
    def calc_peaks(
        predictions: pd.DataFrame, 
        means: pd.DataFrame | None = None, 
        prominence: float = 0.05,
        distance: int = 10,
    ) -> LSTMPeaks:
        
        def subset(df: pd.DataFrame) -> dict[str, Sequence[int]]:
            return {
                name:find_peaks(
                    x=col.index,
                    height=col.to_numpy(),
                    prominence=prominence,
                    distance=distance
                )[0] for name, col in df[__class__.COLUMNS[:-1]].items()
            }
        
        peaks = {feature:subset(df.set_index("time")) for feature, df in predictions.groupby("variable")} 
        if means:
            peaks[None] = subset(means)
        return peaks

    def means(
        self: Self,
        dataset: Dataset
    ) -> pd.DataFrame:
        if dataset not in self.result_means:
            self.result_means[dataset] = self.calc_means(self.results[dataset])
        return self.result_means[dataset]

    def peaks(
        self: Self, 
        dataset: Dataset, 
        prominence: float = 0.05,
        distance: int = 10,
    ) -> LSTMPeaks:
        if dataset not in self.result_peaks:
            self.result_peaks[dataset] = self.calc_peaks(self.results[dataset], self.means(dataset), prominence, distance)
        return self.result_peaks[dataset]
        

    def run(
        self: Self,
        dataset: Dataset,
    ):
        
        self.results[dataset] = pd.concat((self.run_on_series(dataset.df[feature]).reset_index() for feature in dataset.feature_cols.keys()))

    def _print(self: Self, dataset: Dataset):
        preds = self.results[dataset]
        peaks = self.peaks(dataset)
        for col, col_indices in peaks.items():
            df = preds[preds["variable"] == col]
            detected = sum(len(p) for p in col_indices.values())
            if detected != 0:
                print(f"Detected {detected} peaks for {"variable " + col if col is not None else "combined data"}:")
                for type, indices in col_indices.items():
                    for idx in indices:
                        print(f"{df[type].iloc[idx]} ({type}) at {df[type].index[idx]} {dataset.age_format}")
        
    def _plot(self: Self, dataset: Dataset) -> Figure:
                
        fig = plt.figure(figsize=(9, 12))
        fig.suptitle(f"Bifurcation classifications on {dataset.name}")
        
        preds = self.results[dataset]
        
        means = self.means(preds)
        peaks = self.peaks(preds, means)
        
        def plot_peak(ax: Axes, df: pd.DataFrame, col: str):
            indices = peaks[feature][col]
            ax.scatter(df.index[indices], df.iloc[indices])
        
        subplots: Sequence[Axes] = fig.subplots(nrows=len(self.COLUMNS), ncols=1, sharex='all')
        
        mean_ax = subplots[-1]
        mean_ax.invert_xaxis()
        mean_ax.set_xlabel(f"Age ({dataset.age_format})")
        mean_ax.set_ylabel("Mean Feature Probability")
        
        for i, col in enumerate(self.COLUMNS[:-1]): # All besides null column, which
            ax = subplots[i]
            ax.set_ylabel(col + " Probability")
            for feature, data in dataset.features().groupby("variable"):
                data = data.set_index("time")[col]
                ax.plot(data.index, data)
                plot_peak(ax, data, col)
            
            mean_ax.plot(means[col], label=col)
            plot_peak(ax, means, col)
            
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
LABEL_COLS: Final[list[str]] = ['class_label', 'bif', 'null']

type Sims = pd.Series
type Labels = pd.DataFrame
type Groups = pd.Series
type TrainData = tuple[dict[int, Sims], Labels, Groups]

# Returns combined (indexed sims, labels, groups)
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
    
type XY = np.ndarray[np.dtype[float]]
type DeFunc = Callable[[XY, float], XY]

class StochSim():

    def __init__(
        self: Self,
        binit: float,
        bcrit: float,
        ts_len: int,
        dt: float = 0.01,
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
        :type dt: float
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
        sigma: float, 
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
        :type sigma: float
        :param tburn: Burn-in length
        :type tburn: int
        :param clip: Optional clipping function for simulation
        :type clip: Callable[[XY], XY]
        :return: DataFrame of trajectories indexed by time
        :rtype: DataFrame
        """
        
        ## Implement Euler Maryuyama for stocahstic simulation
        s: np.ndarray[np.ndarray[float]] = np.zeros([len(self.parameter),len(s0)])
        
        # Create brownian increments (s.d. sqrt(dt))
        dW_burn = rand.normal(loc=0, scale=np.sqrt(self.dt) * sigma, size = [int(tburn/self.dt),len(s0)])
        dW = rand.normal(loc=0, scale=np.sqrt(self.dt) * sigma, size = [len(self.parameter/self.dt),len(s0)])
        
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
        df_traj = pd.DataFrame({"time": self.parameter.index} | {f"p{i}":s[:,i] for i in range(s.shape[1])})#, 'b': self.parameter.to_numpy()})
        
        # Filter dataframe according to spacing
        df_traj_filt = df_traj.iloc[0::int(self.dt_sample/self.dt)].set_index("time")
        df_traj_filt.index = df_traj_filt.index.astype(int)

        return df_traj_filt