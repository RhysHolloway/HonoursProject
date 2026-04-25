import functools
import itertools
from typing import Any, Final, Self, Sequence, Callable, Iterable, Literal, Tuple, TypeVar

from matplotlib.ticker import ScalarFormatter
from models import Column, Dataset, Model

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from .. import compute_residuals, iter_progress, space_indices

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import logging
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model, Sequential as KerasModel
logging.getLogger("tensorflow").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r"Argument `decay` is no longer supported and will be ignored\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Skipping variable loading for optimizer '.*', because it has .* variables whereas the saved optimizer has .* variables\.",
    category=UserWarning,
)

class LSTMLoader:
    
    def __init__(self: Self, path: str, jobs: int = 4):
        import os
        import os.path
        
        EXTENSIONS = [".keras", ".h5"]

        def iter_model_paths() -> list[str]:
            paths: list[str] = []
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path) and any(file.endswith(ext) for ext in EXTENSIONS):
                        paths.append(file_path)
            return sorted(paths)

        def load_one(file_path: str) -> tuple[int, str, KerasModel]:
            model: KerasModel = load_model(file_path) # type: ignore
            return int(model.input_shape[1]), file_path, model

        found: dict[int, list[tuple[str, KerasModel]]] = dict()
        loaded = Parallel(
            n_jobs=jobs,
            backend="threading",
            prefer="threads",
            require="sharedmem",
        )(delayed(load_one)(file_path) for file_path in iter_model_paths())

        for ts_len, file_path, model in loaded: # type: ignore
            if ts_len not in found:
                found[ts_len] = []
            found[ts_len].append((file_path, model))
        
        if len(found) == 0:
            raise ValueError(f"Could not load any models from {path}!")
        
        self.models: Final[dict[int, list[KerasModel]]] = {
            ts_len:[model for _, model in sorted(models, key=lambda pair: pair[0])]
            for ts_len, models in found.items()
        }
        
    @property
    def with_args(self: Self):
        return functools.partial(LSTM, get_models=self._get_models)
    
    def _get_models(self: Self, length: int) -> list[KerasModel]:
        if length == 0:
            return []
        # Get the closest model length to the input length, preferring smaller models if there is a tie
        return self.models[min(self.models.keys(), key=lambda ts_len:abs(ts_len-length))]

type LSTMPeaks = dict[Column | None, dict[str, np.ndarray]]
class LSTM(Model[pd.DataFrame]):
    
    COLUMNS: Final[list[str]] = ["Fold", "Hopf", "Transcritical", "Null"]
        
    def __init__(
        self: Self,
        get_models: Callable[[int], Sequence[KerasModel]],
        name: str = "LSTM",
        spacing: int = 10,
        window: float | int = 0.25,
        detrend: str = "Lowess",
        verbose: bool = True,
        jobs: int = 4,
    ):
        super().__init__(name)
        
        self.get_models = get_models
        self.verbose = verbose
        self.spacing = spacing
        self.window = window
        self.jobs = jobs
        self.result_means: dict[Dataset, pd.DataFrame] = dict()
        self.result_peaks: dict[Dataset, LSTMPeaks] = dict()
        self.detrend = detrend
    
    def run_on_series(
        self: Self,
        series: pd.Series,
        indices: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        
        models = self.get_models(len(series))
        if len(models) == 0:
            raise ValueError(f"No LSTM models were found to be used with {series.name}!")
        
        if indices is None:
            indices = space_indices(series, self.spacing)

        MODEL_LENGTH = models[0].input_shape[1]
        
        windows: list[tuple[Any, np.ndarray[Any, np.dtype[np.float64]]]] = []
        for imax in indices:
            input = series.iloc[0:imax + 1]
            time = series.index[imax]
            if len(input) == 0:
                continue
            input = input.to_numpy(dtype=float)
            scale = np.mean(np.abs(input))
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            input = input / scale
            
            # Set time series to input shape either by truncation of the start of the series or by appending zeroes to the start
            if len(input) > MODEL_LENGTH:
                input = input[-MODEL_LENGTH:]
            else:
                input = np.concatenate((np.zeros(MODEL_LENGTH - len(input)), input))
            windows.append((time, input))

        if len(windows) == 0:
            return pd.DataFrame(columns=LSTM.COLUMNS + ["time"]).set_index("time")

        times = [time for time, _ in windows]
        inputs = np.stack([data for _, data in windows]).reshape(len(windows), MODEL_LENGTH, 1)
        def compute_model(tsid: int) -> pd.DataFrame:
            model = models[tsid]
            
            # Get DL prediction
            return pd.DataFrame(
                model.predict(
                    inputs, 
                    verbose=0, #type: ignore
                    batch_size=256
                ),
                columns=self.COLUMNS,
                index = pd.MultiIndex.from_arrays(
                    [[tsid] * len(windows), times], 
                    names=["tsid", "time"]
                ),
            )
            
        return self.calc_means(
            pd.concat(
                Parallel(
                    n_jobs=self.jobs,
                    backend="threading",
                    prefer="threads",
                    require="sharedmem",
                    return_as="generator_unordered",
                )(
                    delayed(compute_model)(tsid)
                    for tsid in iter_progress(
                        range(len(models)),
                        self.verbose, 
                        desc=f"{self.name} {series.name}"
                    )
                )
            )
        )
    
    @staticmethod
    def calc_means(predictions: pd.DataFrame) -> pd.DataFrame:
        return predictions[__class__.COLUMNS].dropna().groupby("time").mean()
    
    # @staticmethod
    # def calc_peaks(
    #     predictions: pd.DataFrame, 
    #     means: pd.DataFrame | None = None, 
    #     prominence: float = 0.05,
    #     distance: int = 10,
    # ) -> LSTMPeaks:
        
    #     def subset(df: pd.DataFrame) -> dict[Any, np.ndarray]:
    #         from scipy.signal import find_peaks
    #         return {
    #             name:find_peaks(
    #                 x=col.index.get_level_values("time"),
    #                 height=col.to_numpy(),
    #                 prominence=prominence,
    #                 distance=distance
    #             )[0] for name, col in df[__class__.COLUMNS[:-1]].items()
    #         }
        
    #     peaks: dict[Any, Any] = {feature:subset(df) for feature, df in (predictions.groupby("variable") if "variable" in predictions.index.names else [(None, predictions)])} 
    #     if means is not None:
    #         peaks[None] = subset(means)
    #     return peaks

    def means(
        self: Self,
        dataset: Dataset
    ) -> pd.DataFrame:
        if dataset not in self.result_means:
            if dataset not in self.results:
                self.run(dataset)
            self.result_means[dataset] = self.calc_means(self.results[dataset])
        return self.result_means[dataset]

    # def peaks(
    #     self: Self, 
    #     dataset: Dataset, 
    #     prominence: float = 0.05,
    #     distance: int = 10,
    # ) -> LSTMPeaks:
    #     if dataset not in self.result_peaks:
    #         means = self.means(dataset)
    #         self.result_peaks[dataset] = self.calc_peaks(self.results[dataset], means, prominence, distance)
    #     return self.result_peaks[dataset]

    def run(
        self: Self,
        dataset: Dataset,
        window: float | int | None = None,
        detrend: str | None = None,
    ):
        def compute(feature: Column) -> pd.DataFrame:
            residuals = compute_residuals(dataset.df[feature],span=window or self.window, type=detrend or self.detrend)[::-1] # type: ignore
            df = self.run_on_series(residuals)
            return df.set_index(pd.Series([feature] * len(df), name="variable"), append=True)
        
        self.results[dataset] = pd.concat(map(compute, dataset.df.columns))

    @staticmethod
    def _plot_prob(ax: Axes, feature: Any, col: str, df: pd.DataFrame, label: str | None = None):
        data = df[col]
        ax.plot(data.index.get_level_values("time"), data.to_numpy(), label=label)
        
    @staticmethod                
    def plot(axes: Sequence[Axes], means: pd.DataFrame, reverse: bool = True):
        
        last_ax = axes[-1]
        means_ax = axes[-2]
        
        last_ax.set_xlabel("Age (ya BP)")
        
        last_ax.set_ylim(0, 1)
        means_ax.set_ylim(0, 1)
        
        last_ax.set_ylabel("Probability")
        means_ax.set_ylabel("Probability")

        if reverse:
            last_ax.xaxis.set_inverted(True)
            means_ax.xaxis.set_inverted(True)
        
        means_ax.set_title("Bifurcation Probabilities")
        for col in __class__.COLUMNS[:-1]:
            __class__._plot_prob(means_ax, None, col, means, col)
        means_ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        
        bif_probs = means[__class__.COLUMNS[:-1]].set_index(means.index.get_level_values("time")).sum(axis=1)
        last_ax.set_title("Probability of Any Bifurcation")
        last_ax.plot(bif_probs, label="Bifurcation")
        null = means[__class__.COLUMNS[-1]]
        null.index = null.index.get_level_values("time")
        last_ax.plot(null, label=null.name)
        last_ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )


    def _plot(self: Self, dataset: Dataset, title: bool = True) -> Figure:
        
        preds = self.results[dataset]
        
        multivar = preds is not None and "variable" in preds.index.names and len(preds.index.get_level_values("variable").unique()) > 1
        
        fig, axes = plt.subplots(
            nrows=((len(__class__.COLUMNS) - 1 if multivar else 0) + 2), 
            ncols=1, 
            width_ratios=[3], 
            sharex='all', 
            sharey='all',
        )
        
        __class__.plot(
            axes=axes[0:2],
            means=self.means(dataset), 
            # peaks = self.peaks(dataset)
        )
        
        if multivar:
            axes[0].set_title("Average Bifurcation Probability for all features")
            for i, col in enumerate(__class__.COLUMNS[:-1]): # All besides null column
                ax = axes[i]
                ax.set_title(f"{col} Probability")
                ax.set_xlabel("Age (ya BP)")
                ax.set_ylabel("Probability")
                for feature, data in preds.groupby("variable"): # type: ignore
                    __class__._plot_prob(ax, feature, col, data, str(feature))
                ax.legend()
        
        if title:
            fig.suptitle("Bifurcation classifications on " + dataset.name)
        fig.tight_layout()
        return fig
        

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
        self._parameter_values = self.parameter.to_numpy(copy=False)
        self._time_values = time.to_numpy(dtype=float, copy=False)
        self.dt = dt
        self.dt_sample = dt_sample
        self.sample_step = max(1, int(round(dt_sample / dt)))

        if not np.isclose(self.sample_step * dt, dt_sample):
            raise ValueError(
                f"dt_sample={dt_sample} must be an integer multiple of dt={dt}"
            )
        

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
        
        ## Implement Euler Maryuyama for stochastic simulation
        state = np.array(s0, dtype=float, copy=True)
        n_state = len(state)
        n_steps = len(self._parameter_values)
        sampled_steps = np.arange(0, n_steps, self.sample_step)

        samples: Array = np.empty((len(sampled_steps), n_state))
        samples[:] = np.nan

        dW = rand.normal(
            loc=0,
            scale=np.sqrt(self.dt) * sigma,
            size=(max(0, n_steps - 1), n_state),
        )

        sample_pos = 0
        burn_steps = max(0, int(round(tburn / self.dt)))

        with np.errstate(invalid="raise", over="raise", divide="raise"):
            try:
                if burn_steps > 0:
                    dW_burn = rand.normal(
                        loc=0,
                        scale=np.sqrt(self.dt) * sigma,
                        size=(burn_steps, n_state),
                    )
                    p0 = self._parameter_values[0]
                    for noise in dW_burn:
                        state = state + de_fun(state, p0) * self.dt + noise

                samples[0] = state
                sample_pos = 1

                for i in range(n_steps - 1):
                    state = clip(state + de_fun(state, self._parameter_values[i]) * self.dt + dW[i])
                    if (i + 1) % self.sample_step == 0:
                        samples[sample_pos] = state
                        sample_pos += 1
            except FloatingPointError:
                pass

        return pd.DataFrame(
            samples,
            columns=[f"p{i}" for i in range(n_state)],
            index=pd.Index(self._time_values[sampled_steps], name="time"),
        )
