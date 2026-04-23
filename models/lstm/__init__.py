import functools
import itertools
from typing import Any, Final, OrderedDict, Self, Sequence, Callable, Iterable, Literal, Tuple, TypeVar
from models import Column, Dataset, Model

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from .. import compute_residuals, iter_progress, space_indices

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras.models import load_model, Sequential as KerasModel

class LSTMLoader:
    
    def __init__(self: Self, path: str):
        import os
        import os.path
        
        EXTENSIONS = [".keras", ".h5"]
        
        found: dict[int, list[KerasModel]] = dict()
        
        for path, _, files in os.walk(path):
            for file in files:
                file_path: str = os.path.join(path, file)
                if os.path.isfile(file_path) and any(file.endswith(ext) for ext in EXTENSIONS):
                    model: KerasModel = load_model(file_path) # type: ignore
                    ts_len = model.input_shape[1]
                    if ts_len not in found:
                        found[ts_len] = list()
                    found[ts_len].append(model)
        
        if len(found) == 0:
            raise ValueError(f"Could not load any models from {path}!")
        
        self.models: Final[dict[int, list[KerasModel]]] = found
        self.keys: Final[list[int]] = sorted(self.models.keys())
        
    @property
    def with_args(self: Self):
        return functools.partial(LSTM, get_models=self._get_models)
    
    def _get_models(self: Self, length: int) -> list[KerasModel]:
        if length == 0:
            return []
        # Get the model with the closest input length lower or equal to the input series length (or else get the model with the lowest input length if none can be found)
        ts_len = min((ts_len for ts_len in self.keys if ts_len >= length), default=self.keys[-1])
        # print("Getting models for", ts_len, "for series of length", length)
        return self.models[ts_len]

type LSTMPeaks = dict[Column | None, dict[str, np.ndarray]]
class LSTM(Model[pd.DataFrame]):
    
    COLUMNS: Final[list[str]] = ["Fold", "Hopf", "Branch", "Null"]
        
    def __init__(
        self: Self,
        get_models: Callable[[int], Sequence[KerasModel]],
        name: str = "LSTM",
        spacing: int = 10,
        verbose: bool = True,
    ):
        super().__init__(name)
        
        self.get_models = get_models
        self.verbose = verbose
        self.spacing = spacing
        self.result_means: dict[Dataset, pd.DataFrame] = dict()
        self.result_peaks: dict[Dataset, LSTMPeaks] = dict()
    
    def run_on_series(
        self: Self,
        series: pd.Series,
        parallel: Parallel,
        indices: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        
        models = self.get_models(len(series))
        if len(models) == 0:
            raise ValueError(f"No LSTM models were found to be used with {series.name}!")
        
        if indices is None:
            indices = list(range(len(series)))[::self.spacing]
        
        if len(indices) == 0:
            return pd.DataFrame(columns=LSTM.COLUMNS + ["time"]).set_index("time")
        
        start = indices[0]
        indices = indices[1:]
                
        def compute(tsid: int, imax: int) -> pd.DataFrame:
            model = models[tsid]
            input = series.iloc[start:imax]
            if len(input) == 0:
                return pd.DataFrame(columns=self.COLUMNS + ["tsid", "time"]).set_index(["tsid", "time"])
            time = input.index.get_level_values("time")[-1]
            input = input.to_numpy()
            input = input / np.mean(np.abs(input))
            
            # Set time series to input shape either by truncation of the start of the series or by appending zeroes to the start
            if len(input) > model.input_shape[1]:
                input = input[-model.input_shape[1]:]
            else:
                input = np.concatenate((np.zeros(model.input_shape[1] - len(input)), input))

            # Get DL prediction
            return pd.DataFrame(
                model.predict(
                    input.reshape(1, -1, 1), 
                    verbose=0, #type: ignore
                    batch_size=256
                    ), 
                columns=self.COLUMNS, 
                index = pd.MultiIndex.from_tuples([(tsid, time)], names=["tsid", "time"])
            )
            
        return self.calc_means(
            pd.concat(
                parallel(
                    delayed(compute)(tsid, imax) 
                    for tsid, imax in iter_progress(
                        itertools.product(range(len(models)), indices),
                        self.verbose, 
                        desc=f"{self.name} {series.name}"
                    )
                )
            )
        )
    
    @staticmethod
    def calc_means(predictions: pd.DataFrame) -> pd.DataFrame:
        return predictions[__class__.COLUMNS].dropna().groupby("time").mean()
    
    @staticmethod
    def calc_peaks(
        predictions: pd.DataFrame, 
        means: pd.DataFrame | None = None, 
        prominence: float = 0.05,
        distance: int = 10,
    ) -> LSTMPeaks:
        
        def subset(df: pd.DataFrame) -> dict[Any, np.ndarray]:
            from scipy.signal import find_peaks
            return {
                name:find_peaks(
                    x=col.index.get_level_values("time"),
                    height=col.to_numpy(),
                    prominence=prominence,
                    distance=distance
                )[0] for name, col in df[__class__.COLUMNS[:-1]].items()
            }
        
        peaks: dict[Any, Any] = {feature:subset(df) for feature, df in predictions.groupby("variable")} 
        if means is not None:
            peaks[None] = subset(means)
        return peaks

    def means(
        self: Self,
        dataset: Dataset
    ) -> pd.DataFrame:
        if dataset not in self.result_means:
            if dataset not in self.results:
                self.run(dataset)
            self.result_means[dataset] = self.calc_means(self.results[dataset])
        return self.result_means[dataset]

    def peaks(
        self: Self, 
        dataset: Dataset, 
        prominence: float = 0.05,
        distance: int = 10,
    ) -> LSTMPeaks:
        if dataset not in self.result_peaks:
            means = self.means(dataset)
            self.result_peaks[dataset] = self.calc_peaks(self.results[dataset], means, prominence, distance)
        return self.result_peaks[dataset]

    def run(
        self: Self,
        dataset: Dataset,
    ):
        parallel = Parallel(return_as="generator_unordered", backend="threading", n_jobs=4)
        
        def compute(feature: Column) -> pd.DataFrame:
            residuals = compute_residuals(dataset.df[feature])[::-1] # type: ignore
            df = self.run_on_series(residuals, parallel=parallel)
            df["variable"] = np.array([feature] * len(df))
            return df.set_index("variable", append=True)
        
        self.results[dataset] = pd.concat(map(compute, dataset.df.columns))

    def _print(self: Self, dataset: Dataset):
        preds = self.results[dataset]
        peaks = self.peaks(dataset)
        for col, col_indices in peaks.items():
            df = preds[preds.index.get_level_values("variable") == col]
            detected = sum(len(p) for p in col_indices.values())
            if detected != 0:
                print(f"Detected {detected} peaks for {f'variable {col}' if col is not None else 'combined data'}:")
                for type, indices in col_indices.items():
                    for idx in indices:
                        print(f"{df[type].iloc[idx]} ({type}) at {df[type].index[idx]} {dataset.age_format}")

    @staticmethod                
    def plot(name: str, legend: Iterable[str] | None = None, age: str | None = None, preds: pd.DataFrame | None = None, means: pd.DataFrame | None = None, peaks: LSTMPeaks | None = None, reverse: bool = True) -> Figure:
                
        if means is None and preds is None:
            raise ValueError("Please provide either the means or the predictions to plot for the LSTM!")
                
        fig = plt.figure(figsize=(9, 1 + (5 if means is not None else 0) + (8 if preds is not None else 0)))
        fig.suptitle(f"Bifurcation classifications on {name}")
        
        AGE = f"Age ({age})" if age is not None else "Age"
        BIF_COLS = __class__.COLUMNS[:-1]
        
        subplots: Sequence[Axes] = fig.subplots(nrows=((len(BIF_COLS) if preds is not None else 0) + (2 if means is not None else 0)), ncols=1, sharex='all', sharey='all')
        
        last_ax = subplots[-1]
        last_ax.set_xlabel(AGE)
        last_ax.set_ylim(0, 1)
        if reverse:
            last_ax.invert_xaxis()
        
        def prob_plot(ax: Axes, feature: Any, col: str, df: pd.DataFrame, label: str | None = None):
            data = df[col]
            data.index = data.index.get_level_values("time")
            ax.plot(data, label=label)
            if peaks is not None:
                indices = peaks[feature][col]
                ax.scatter(x=data.index[indices], y=data.iloc[indices].to_numpy())
        
        if means is not None:
            means_ax = subplots[-2]
            means_ax.set_ylabel("Mean Feature Probability")
            for i, col in enumerate(BIF_COLS):
                prob_plot(means_ax, None, col, means, col)
            means_ax.legend()
            
            bif_probs = means[BIF_COLS].set_index(means.index.get_level_values("time")).sum(axis=1)
            last_ax.set_ylabel("Combined Bifurcation Probability")
            last_ax.plot(bif_probs, label="Combined")
            null = means[__class__.COLUMNS[-1]]
            null.index = null.index.get_level_values("time")
            last_ax.plot(null, label=null.name)
            last_ax.legend()
        
        if preds is not None:
            for i, col in enumerate(BIF_COLS): # All besides null column, which
                ax = subplots[i]
                ax.set_xlabel(AGE)
                ax.set_ylabel(col + " Probability")
                for feature, data in preds.groupby("variable"):
                    prob_plot(last_ax, feature, col, data, col)
            if legend is not None:
                fig.legend(legend)
        
        return fig

    def _plot(self: Self, dataset: Dataset) -> Figure:
        return __class__.plot(
            name = dataset.name,
            legend = dataset.feature_names(),
            age = Dataset.age_format(dataset.df.index),
            preds = self.results[dataset], 
            means = self.means(dataset), 
            peaks = self.peaks(dataset)
        )

type BifId = Literal['HB', 'BP', 'LP']
type BifType = Tuple[BifId, bool]
BIFS: Final[list[BifId]] = ["HB", "BP", "LP"]

def bif_types():
    for null in [True, False]:
        for type in BIFS:
            yield (type, null)

def bif_maximum(type: BifType, bif_max: int) -> int:
    id, null = type
    if null:
        return ((bif_max - 2 * (bif_max // 3)) if (id == "BP") else (bif_max // 3))
    else:
        return bif_max

INDEX_COL: Final[str] = 'sequence_ID'
LABEL_COLS: Final[list[str]] = ['class_label', 'bif', 'null']

type Sims = pd.Series
type Labels = pd.DataFrame
type Groups = pd.DataFrame
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
    
    return functools.reduce(_reduce_combine, batches)

type Array = np.ndarray[Any, np.dtype[np.float64]]
type DeFunc = Callable[[Array, float], Array]
    

class StochSim():

    def __init__(
        self: Self,
        b_start: float,
        b_end: float,
        t_end: int,
        dt: float = 0.01,
        dt_sample: float = 1.0,
        t_start: int = 0,
    ):
        """
        Stochastic simulator class
        
        :param self: Simulator class
        :type self: Self
        :param b_start: Initial value of bifurcation parameter
        :type b_start: float
        :param b_end: Bifurcation point of model
        :type b_end: float
        :param t_end: End time in time series
        :type t_end: int
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
        :param t_start: Start time in time series
        :type t_start: int
        """
        
        # Initialise arrays to store single time-series data
        time = pd.Index(np.arange(t_start, t_end, dt), name="time")
        
        # Set up bifurcation parameter b, that increases linearly in time from binit to bcrit
        self.parameter = pd.Series(np.linspace(b_start,b_end,len(time)),index=time)
        
        self.dt = dt
        self.dt_sample = dt_sample
        

    # Throws floating point error
    def simulate(
        self: Self,
        de_fun: DeFunc,
        s0: Array,
        sigma: Array | float, 
        tburn: int = 100, # burn-in period
        clip: Callable[[Array], Array] = lambda s: s,
        rand: np.random.RandomState = np.random.mtrand._rand,
    ) -> pd.DataFrame:      
        """
        Function to run a stochastic simulation of model up to bifurcation point
        
        :param self: Initialized simulation class
        :type self: Self
        :param de_fun: Derivative equation to simulate
        :type de_fun: DeFunc
        :param s0: Initial value
        :type s0: Array
        :param sigma: amplitude factor of GWN - total amplitude also depends on parameter values
        :type sigma: float
        :param tburn: Burn-in length
        :type tburn: int
        :param clip: Optional clipping function for simulation
        :type clip: Callable[[Array], Array]
        :return: DataFrame of trajectories indexed by time
        :rtype: DataFrame
        """
        
        ## Implement Euler Maryuyama for stocahstic simulation
        s: Array = np.empty([len(self.parameter),len(s0)])
        s[:] = np.nan
        
        # Create brownian increments (s.d. sqrt(dt))
        dW = rand.normal(loc=0, scale=np.sqrt(self.dt) * sigma, size = [len(self.parameter/self.dt),len(s0)])
        
        # Run burn-in period on s0
        with np.errstate(invalid="raise", over="raise", divide="raise"):
            try:
                if tburn > 0:
                    dW_burn = rand.normal(loc=0, scale=np.sqrt(self.dt) * sigma, size = [int(tburn/self.dt),len(s0)])
                    # Get initial condition post burn-in period
                    for i in range(int(tburn/self.dt)):
                        s0 = s0 + de_fun(s0, self.parameter.iloc[0]) * self.dt + dW_burn[i]
                    s[0] = s0
                
                # Run simulation, updating bifurcation parameter
                for i in range(len(self.parameter)-1):
                    s[i+1] = clip(s[i] + de_fun(s[i], self.parameter.iloc[i])*self.dt + dW[i])
            finally:
                df = pd.DataFrame(s, columns=[f"p{i}" for i in range(s.shape[1])], index=self.parameter.index).iloc[0::int(self.dt_sample/self.dt)]
                df.index = df.index.astype(int)
                return df