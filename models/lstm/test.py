import abc
import functools
import itertools
import traceback
from typing import Any, Callable, Final, Iterable, Literal, Self, Sequence, Tuple
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os.path

import models.lstm as lstm_module
import models.metrics as metrics_module
from models import Dataset, compute_residuals, iter_progress, space_indices
from models.metrics import Metrics
from models.lstm import LSTMLoader, LSTM, StochSim, Array, DeFunc

type ModelData = tuple[pd.Series, pd.Series, float]
    

class TestModel(metaclass = abc.ABCMeta):
        
    INDICES: Final[list[str]] = ["forced", "tsid", "time"]
    COLUMNS: Final[list[str]] = ["state", "residuals"]
    
    def __init__(
        self: Self,
        name: str,
        window: float | int = 0.25,
        sims: int = 10,
    ):
        self.name = name
        self.sims = sims
        self.window = window
        
    # Returns residuals
    @abc.abstractmethod
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> dict[str, pd.DataFrame]:
        """
        Method to generate/load and return multiple variations of simulations (and possibly existing data) to run metrics on (for example, different lengths for test models and slices of data trimmed to end at tipping points for datasets)
        
        :param self: Simulator class
        :type self: Self
        :param parallel: Enables computing simulations in parallel
        :type parallel: Parallel
        :param verbose: Whether to print progress to output
        :type verbose: bool
        :param path: Location to load/save simulations
        :type path: str | None
        """
        pass
    
    def load(self: Self, path: str | None, file: str, /, indices: list[str] = INDICES, columns: list[str] = COLUMNS, checks: Callable[[pd.DataFrame], pd.DataFrame | None] = lambda df: df, time: float | None = None) -> pd.DataFrame | None:
        if path is None:
            return None
        try:
            FILE_PATH = os.path.join(path, self.name, file)
            if time is not None and os.path.exists(FILE_PATH) and os.path.getmtime(FILE_PATH) < time:
                return None
            df = pd.read_csv(FILE_PATH, index_col=indices)[columns]
            return checks(df)
        except FileNotFoundError:
            return None
        except:
            import traceback
            import sys
            traceback.print_exc(limit = 0)
            print(f"Error loading {file} for model {self.name} from path {path}.", file=sys.stderr)
            return None
        
    def save(self: Self, path: str | None, file: str, df: pd.Series | pd.DataFrame, /):
        if path is not None:
            path = os.path.join(path, self.name)
            os.makedirs(path, exist_ok=True)
            df.to_csv(os.path.join(path, file))
    
    def metrics(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        residuals = df["residuals"]
        rolling = residuals.rolling(window=Metrics.window_index(residuals, self.window))
        return pd.concat(
            itertools.chain([df], 
                (
                    func(rolling)
                    for func in [Metrics.variance, Metrics.ac1]
                )
            ), axis=1)
    
    @staticmethod
    def add_indices(df: pd.Series | pd.DataFrame, forced: bool | np.bool, tsid: int | np.integer):
        df.index = pd.MultiIndex.from_arrays(
            [
                np.array([forced] * len(df)), 
                np.array([tsid] * len(df)),
                df.index, 
            ], 
            names=__class__.INDICES
        )
        

class SimModel(TestModel):
    
    FILE: Final[str] = "sims.csv"
    
    def __init__(
        self: Self,
        name: str,
        de_fun: DeFunc,
        bl: float, 
        bh: float, 
        bcrit: float,
        init: Array,
        sigma: Array | float,
        lengths: set[int],
        sims: int = 10,
    ):
        super().__init__(
            name = name,
            sims = sims,
        )
        
        self.de_fun = de_fun
        self.bl = bl
        self.bh = bh
        self.bcrit = bcrit
        self.init = init
        self.sigma = sigma
        
        self.lengths = lengths
        
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> dict[str, pd.DataFrame]:
        
        max_length = max(self.lengths)
        
        def filter_too_short(df: pd.DataFrame):
            return df.groupby(by=__class__.INDICES[:-1], dropna=False).filter(lambda group: group.index.get_level_values("forced").to_numpy(bool).any() or np.ceil(len(group)) >= max_length)
        
        load = self.load(path, __class__.FILE, checks=filter_too_short)
        df = pd.DataFrame(columns=self.INDICES + __class__.COLUMNS).set_index(self.INDICES) if load is None else load
                
        remaining = set(itertools.product(range(self.sims), [True, False])).difference(iter(df.groupby(["tsid", "forced"]).groups.keys()))
        # if verbose and len(remaining) != 0:
        #     print("Missing simulations", remaining)
        if len(remaining) > 0:
            
            df: pd.DataFrame = pd.concat(
                iter_progress(
                    itertools.chain(
                        [df],
                        parallel(
                            delayed(self._simulate_once)(max_length, forced, tsid)
                            for tsid, forced in remaining
                        )), verbose, desc=self.name + " simulations"))
            
            self.save(path, __class__.FILE, df)

        return {f"len{length}":pd.concat(self.metrics(df.iloc[-length:]) for _, df in df.groupby(__class__.INDICES[:-1])) for length in self.lengths} # type: ignore        
    
    def _simulate_once(
        self: Self,
        length: int,
        forced: np.bool,
        tsid: int,
    ) -> pd.DataFrame:

        model = StochSim(
            b_start=self.bl,
            b_end=self.bh if forced else self.bl,
            t_end=length,
        )
        
        series = model.simulate(
            de_fun=self.de_fun,
            s0=self.init,
            sigma=self.sigma,
            clip=lambda xy: np.clip(xy, a_min=0, a_max=None)
        )["p0"]
        
        if forced:
            tcrit = model.parameter[model.parameter > self.bcrit].index[0]
            series = series[series.index <= tcrit]
            
        df = pd.concat([
                    series.to_frame(name="state"), 
                    compute_residuals(series)
                    ], axis=1)
            
        self.add_indices(df, forced, tsid)
        
        return df


class MayFold(SimModel):
        
    def __init__(self: Self, lengths: set[int]):    
                
        K: Final[float] = 1  # carrying capacity
        S: Final[float] = 0.1  # half-saturation constant of harvesting function
        
        def de_fun(x: Array, a: float) -> Array:
            x = x[0]
            return np.array([x * (1 - x / K) - a * (x**2 / (S**2 + x**2))])
        
        super().__init__(
            name="may_fold",
            de_fun=de_fun,
            bl=0.15,
            bh=0.27,
            bcrit = 0.260437,
            init=np.array([0.8197]),
            sigma=0.01,
            lengths=lengths,
        )

class CrModel(SimModel):
    
    def __init__(self: Self, name: str, al: float, ah: float, abif: float, lengths: set[int], sf: int = 4):
        """
        Consumer-resource model
        """
                
        # Model parameters
        R: Final[float] = 1 * sf # Per capita growth rate
        K: Final[float] = 1.7 # Carrying capacity
        H: Final[float] = 0.6 / sf # Consumer attack rate
        E: Final[float] = 0.5 # Conversion factor
        M: Final[float] = 0.5 * sf # Per capita consumer mortality rate
        
        def de_fun(s: Array, a: float) -> Array:
            x, y = s
            p = (a * x * y) / (1 + a * H * x)
            return np.array([
                R * x * (1 - x / K) - p,
                E * p - M * y
            ])
            
        super().__init__(
            name=name,
            de_fun=de_fun,
            bl=al * sf,
            bh=ah * sf,
            bcrit=abif * sf,
            init=np.array([1, 0.412]),
            sigma=0.01,
            lengths=lengths,
        )
    
class CrTrans(CrModel):  

    def __init__(self: Self, lengths: set[int], sf: int = 4):
        super().__init__(
            name="cr_trans",
            al=0.5,
            ah=1.5,
            abif=1.4,
            lengths=lengths,
            sf=sf,
        )
    
class CrHopf(CrModel):
    def __init__(self: Self, lengths: set[int], sf: int = 4):
            
        super().__init__(
            name="cr_hopf",
            al=3,
            ah=4,
            abif=3.923,
            lengths=lengths,
            sf=sf,
        )
        
class DatasetModel(TestModel):
    
    def __init__(
        self: Self,
        dataset: Dataset,
        variable: str,
        tcrit: int | float,
        bandwidth: int | float,
        window: int | float = 0.25,
        detrend: str | None = None,
        sims: int = 10,
    ):
        super().__init__(
            name=dataset.name,
            window=window,
            sims=sims,
        )

        # Get times prior to transition (in years ago)
        prior = dataset.df.loc[dataset.df.index >= tcrit, variable].astype(float).dropna()[::-1]
        prior.index = -prior.index.astype(float)

        # Make dataframe for interpolated data
        df_inter = pd.DataFrame(
            {"keep": True}, 
            dtype=bool,
            index=pd.Index(np.linspace(prior.index[0], prior.index[-1], len(prior)), name=prior.index.name),
        )
        
        # Concatenate with original, interpolate, and extract the interpolated data
        interpolated = pd.concat([prior, df_inter]).infer_objects()
        interpolated[variable] = interpolated[variable].interpolate(method="index")
        self.series = interpolated.dropna(subset="keep")[variable]
        
        self.bandwidth = float(bandwidth)
        self.tcrit = tcrit
        self.detrend = detrend or "Lowess"
        
    def compute(self: Self, series: pd.Series) -> pd.DataFrame:
        residuals = compute_residuals(series, span=self.bandwidth, type=self.detrend) # type: ignore
        return self.metrics(pd.concat([series.to_frame("state"), residuals], axis = 1))
    
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> dict[str, pd.DataFrame]:
        
        COLUMNS = __class__.COLUMNS + ["variance", "ac1"]
        NAME = f"tcrit{np.round(self.tcrit, decimals=2)}"
        FILE = NAME + "_sims.csv"

        df_load = self.load(path, FILE, columns = COLUMNS)
        if df_load is not None:
            groups = {
                (bool(forced), int(tsid)): group.index.get_level_values("time") # type: ignore
                for (forced, tsid), group in df_load.groupby(__class__.INDICES[:-1], dropna=False)
            }

            cache_valid = groups.get((True, 0), pd.Index([])).equals(self.series.index) and all(
                groups.get((False, tsid), pd.Index([])).equals(self.series.index)
                for tsid in range(self.sims)
            )

            if cache_valid:
                return {NAME:df_load}

        df = self.compute(self.series)
        self.add_indices(df, True, 0)

        # Generate inputs for null simulation based on beginning of time-series
        def map_null(group: pd.DataFrame):
            residuals = group["residuals"].dropna()
            window = residuals[:len(residuals) // 5]
        
            alpha = window.autocorr(lag=1)
            sigma: float = np.sqrt(window.var() * (1.0 - alpha**2.0))
            x0: float = residuals.iloc[0]  # initial condition   
            time = group.index.get_level_values("time")
            
            return alpha, sigma, x0, time
        
        # Generate a null simulation DataFrame given inputs
        def compute_null(alpha: float, sigma: float, x0: float, tsid: int, time: pd.Index):
            length = len(time)
            dW = np.random.normal(size=length - 1)
            s = np.empty(length)
            s[:] = np.nan
            s[0] = x0

            # Run recursion in time
            for i in range(1, length):
                # Generate noise increment N(0,1)
                s[i] = alpha * s[i-1] + sigma * dW[i - 1]
                
            df = self.compute(pd.Series(s, index=time))
            self.add_indices(df, False, tsid)
            return df

        # Generate N null simulations
        df = pd.concat(
            iter_progress(itertools.chain(
                [df], 
                parallel(
                    delayed(compute_null)(alpha, sigma, x0, tsid, time) 
                    for (alpha, sigma, x0, time), tsid in 
                    itertools.product(
                        [map_null(df)], 
                        range(self.sims)
                ))), verbose, self.name + " simulations"))
        
        self.save(path, FILE, df)
        
        return {NAME:df}
            
    
def roc(
    model: TestModel,
    kind: str,
    df: pd.DataFrame, 
    lstm: LSTM | None = None,
    path: str | None = None,
    bifurcation: Literal["All", "Hopf", "Fold", "Transcritical"] = "All",
) -> pd.DataFrame:
    
    LEVELS = ["i", "name", "start", "end"]
        
    FILE = f"roc_{kind}.csv"
            
    load = model.load(path, FILE, indices=LEVELS, columns=["fpr", "tpr", "thresholds", "auc"])
    if load is not None:
        return load
    
    df = df.reorder_levels(TestModel.INDICES)

    def roc_interval(df: pd.DataFrame, interval: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]) -> pd.DataFrame:
        df = df.sort_index(level="time")
        group_time = df.index.get_level_values("time")
        forced_df = df.loc[df.index.get_level_values("forced").to_numpy(dtype=bool)]
            
        def per_var(series: pd.Series, name: str) -> pd.DataFrame:
            time = series.index.get_level_values("time")
            t_start = group_time[0]
            t_end = forced_df.reset_index().groupby("tsid")["time"].max().min() if len(forced_df) != 0 else group_time[-1]

            # Get prediction interval in time
            t_pred = t_start + (t_end - t_start) * interval

            # Get data within prediction interval
            trans_series: pd.Series = series[((time >= t_pred[0]) & (time <= t_pred[1]))].dropna()
            truth = trans_series.index.get_level_values("forced").astype(int)
            if len(trans_series) == 0 or len(np.unique(truth)) < 2:
                return pd.DataFrame(columns=["fpr", "tpr", "thresholds", "auc", "name"]).set_index("name")
        
            # Compute ROC curve and threhsolds using sklearn
            fpr, tpr, thresholds = metrics.roc_curve(truth, trans_series.to_numpy())

            df = pd.DataFrame({
                "fpr": fpr, 
                "tpr": tpr, 
                "thresholds": thresholds, 
                "auc": metrics.auc(fpr, tpr), # Compute AUC (area under curve)
                "name": name,
            })
            
            df.index.name = "i"
            return df.set_index("name", append=True)
        
        ROC_TYPES: dict[str, str] = {
            "ktau_variance": "Variance",
            "ktau_ac1": "Lag-1 AC",
        }
            
        frames = [per_var(df[col], name) for col, name in ROC_TYPES.items() if col in df]
            
        if lstm is not None and set(LSTM.COLUMNS).issubset(df.columns):
            combined = df[LSTM.COLUMNS[:-1]].sum(axis=1, min_count=len(LSTM.COLUMNS[:-1])) if bifurcation == "All" else df[bifurcation]
            frames.append(per_var(combined, lstm.name))

        frames = [frame for frame in frames if len(frame) != 0]
        if len(frames) == 0:
            return pd.DataFrame(columns=["fpr", "tpr", "thresholds", "auc", "name", "start", "end"]).set_index(["name", "start", "end"])

        preds = pd.concat(frames)
            
        preds["start"] = interval[0]
        preds["end"] = interval[1]

        return preds.set_index(["start", "end"], append=True)
    
    interval_frames = [roc_interval(df, interval) for interval in [np.array([0.6, 0.8]), np.array([0.8, 1])]]
    interval_frames = [frame for frame in interval_frames if len(frame) != 0]
    if len(interval_frames) == 0:
        df = pd.DataFrame(columns=["fpr", "tpr", "thresholds", "auc"], index=pd.MultiIndex.from_tuples([], names=LEVELS))
    else:
        df = pd.concat(interval_frames).reorder_levels(LEVELS)

    model.save(path, FILE, df)
    
    return df

def counts(df: pd.DataFrame) -> pd.DataFrame:
    if set(LSTM.COLUMNS).issubset(df.columns):
        counts = df.loc[(df[LSTM.COLUMNS].notna().all(axis=1) & df.index.get_level_values("forced").to_numpy()), LSTM.COLUMNS].idxmax(axis=1).value_counts()
        return pd.DataFrame({col.lower():[counts.get(col, 0)] for col in LSTM.COLUMNS})
    else:
        return pd.DataFrame({col.lower():0 for col in LSTM.COLUMNS})

type _ModelGetter = Tuple[str, list[ModelData] | Callable[[], list[ModelData]]]

def test_models(lengths: set[int]) -> list[TestModel]:
    return [MayFold(lengths), CrHopf(lengths), CrTrans(lengths)]
        
def get_metrics(model: TestModel, lstm: LSTM | None = None, path: str | None = None, verbose: bool = True) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    
    parallel = Parallel(return_as="generator_unordered", backend="threading", n_jobs=4)
    cache_sources = [__file__, metrics_module.__file__, lstm_module.__file__]
    
    MODIFIED_TIME = max(
        os.path.getmtime(source)
        for source in cache_sources
        if source is not None and os.path.exists(source)
    )
    
    def per_sim(kind: str, forced: np.bool, tsid, df: pd.DataFrame) -> pd.DataFrame:
        assert pd.api.types.is_integer(tsid)

        FILE = f"metrics_{"forced" if forced else "null"}_{kind}_{tsid}.csv"
        COLUMNS = ["ktau_variance", "ktau_ac1"] + (LSTM.COLUMNS if lstm is not None else [])
        df = df.reset_index(TestModel.INDICES[:-1], drop=True).sort_index()

        indices = space_indices(df["residuals"], spacing=10)

        metrics = model.load(
            path, FILE, 
            columns=COLUMNS, 
            # time=MODIFIED_TIME,
        )
        
        if metrics is None:

            metrics = pd.concat(
                (
                    Metrics.ktau(df[col], indices=indices).to_frame(name=f"ktau_{col}")
                    for col in ["variance", "ac1"]
                ),
                axis=1,
            )

            if lstm is not None:
                metrics = lstm.run_on_series(df["residuals"], indices=indices).join(metrics, how = "outer")

            TestModel.add_indices(metrics, forced, tsid)
            
            metrics = metrics.sort_index()
            
            model.save(path, FILE, metrics)
        
        return metrics
            
    metrics_dict = {
        kind:(full_df, pd.concat(
            iter_progress((per_sim
                (kind, np.bool(forced), tsid, df) 
                for (forced, tsid), df 
                in full_df.groupby(TestModel.INDICES[:-1], dropna=False)
            ), verbose=verbose, desc=model.name))
        ) for kind, full_df in model.simulate(parallel, verbose=True, path=path).items()
    }
    
    return {kind:(sims, df, roc(
                model=model,
                kind=kind,
                df=df,
                lstm=lstm,
            )) for kind, (sims, df) in metrics_dict.items()}
    

def plot_metrics(model: TestModel, kind: str, sims: pd.DataFrame, metrics: pd.DataFrame, roc: pd.DataFrame, output: str, lstm: LSTM | None = None):
    
    groups = roc.groupby(["start", "end"])
    roc_fig, roc_axes = plt.subplots(
        nrows=1,
        ncols=groups.ngroups,
        figsize=(4.0 * groups.ngroups, 5),
        squeeze=False,
        constrained_layout=True,
    )
    
    roc_axes = roc_axes.flatten()
        
    for axis, ((start, end), roc) in zip(roc_axes, groups):
        axis: Axes
        
        assert pd.api.types.is_float(start)
        assert pd.api.types.is_float(end)
        
        groupby = roc.groupby("name")
        
        for name, roc in groupby:
            axis.plot(
                roc["fpr"],
                roc["tpr"],
                label=f"{name}\n(AUC={np.round(roc["auc"].iloc[0], 2)})"
            )

        # Line y=x
        axis.plot(
            [0,1],
            [0,1],
            color="black", 
            linestyle="dashed",
        )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=groupby.ngroups,
            fancybox=True,
            shadow=True,
            fontsize="small",
        ) 
        axis.set_xlabel("False positive rate")
        axis.set_xbound(-0.01, 1)
        axis.set_ylabel("True positive rate") 
        axis.set_aspect('equal', adjustable='box')
        axis.set_title(f"Window of {start * 100}% - {end * 100}% of data)")
                   
    # fig.suptitle()
    # fig.tight_layout()
    
    output = os.path.join(output, model.name)
    os.makedirs(output, exist_ok=True)
    roc_fig.savefig(os.path.join(output, f"ROC {model.name} {kind}.png"))
    
    plt.close(roc_fig)
    
    def get_plots() -> tuple[Figure, Sequence[Axes]]:
        rows = 3 + (2 if lstm is not None else 0)
        return plt.subplots(nrows = rows, sharex=True, figsize=(5, 10)) 
    
    forced_fig, forced_axes = get_plots()
    null_fig, null_axes = get_plots()
    
    for forced, df in sims.groupby("forced", dropna=False):
        
        axes = forced_axes if forced else null_axes
        
        means = df.groupby("time").mean()
        means.index = means.index.get_level_values("time")
        
        axes[0].plot(means["state"], alpha = 0.7)
        axes[0].plot(means["state"] - means["residuals"])
        axes[0].set_ylabel("State")
        Metrics.plot(axes[1:3], means)
    
    if lstm is not None:
        for forced, df in metrics.groupby("forced", dropna=False):
            
            axes = forced_axes if forced else null_axes
            
            lstm.plot(
                axes = axes[3:5],
                means = lstm.calc_means(df),
                reverse = False,
            )
            
    forced_fig.tight_layout()
    forced_fig.savefig(os.path.join(output, f"Forced {model.name} {kind}.png"))
    plt.close(forced_fig)
    
    null_fig.tight_layout()
    null_fig.savefig(os.path.join(output, f"Null {model.name} {kind}.png"))
    plt.close(null_fig)
            

def load_and_save(models: Iterable[TestModel], output: str, lstm: LSTM | None = None, metrics_path: str | None = None):
    for model in models:
        print("Generating/Loading metrics for", model.name)
        try:
            metrics = get_metrics(lstm = lstm, path = metrics_path, model = model)
            print("Plotting metrics for", model.name)
            for kind, (sims, metrics, roc) in metrics.items():
                plot_metrics(model, kind=kind, sims=sims, metrics=metrics, roc=roc, output=output, lstm=lstm)
        except:
            traceback.print_exc()
            import sys
            print("Could not generate metrics for ", model.name, file=sys.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Tester',
                    description='Tests trained models against simulated data')
    parser.add_argument('output', type=str)
    parser.add_argument('--lstm', '-l', type=str)
    parser.add_argument('--simulations', '-s', type=str)
    parser.add_argument('--metrics', '-i', type=str)
    args = parser.parse_args()
    
    load_and_save(
        metrics_path=args.metrics, 
        output=args.output, 
        lstm=LSTMLoader(args.lstm).with_args(verbose=False) if args.lstm is not None else None,
        models=test_models(set([1500, 500])),
    )
