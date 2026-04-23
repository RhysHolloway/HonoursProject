import abc
import functools
import itertools
import traceback
from typing import Any, Callable, Final, Iterable, Literal, Self, Sequence, Tuple
from joblib import Parallel, delayed
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os.path

from models import Dataset, compute_residuals, interpolate, iter_progress, space_indices
from models.metrics import Metrics
from models.lstm import LSTMLoader, LSTM, StochSim, Array, DeFunc

type ModelData = tuple[pd.Series, pd.Series, float]
    

class TestModel(metaclass = abc.ABCMeta):
        
    INDICES: Final[list[str]] = ["variable", "forced", "tsid", "time"]
    COLUMNS: Final[list[str]] = ["state", "residuals"]
    
    def __init__(
        self: Self,
        name: str,
        spacing: int,
        age: str | None,
        sims: int = 10,
    ):
        self.name = name
        self.spacing = spacing
        self.age = age
        self.sims = sims
        
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
    
    def load(self: Self, path: str | None, file: str, /, indices: list[str] = INDICES, columns: list[str] = COLUMNS, checks: Callable[[pd.DataFrame], pd.DataFrame | None] = lambda df: df) -> pd.DataFrame | None:
        if path is None:
            return None
        try:
            df = pd.read_csv(os.path.join(path, self.name, file), index_col=indices)[columns]
            return checks(df)
        except:
            import traceback
            traceback.print_exc(limit = 0)
            return None
        
    def save(self: Self, path: str | None, file: str, df: pd.Series | pd.DataFrame, /):
        if path is not None:
            path = os.path.join(path, self.name)
            os.makedirs(path, exist_ok=True)
            df.to_csv(os.path.join(path, file))
    
    def metrics(self: Self, df: pd.DataFrame) -> pd.DataFrame:
        residuals = df["residuals"]
        rolling = residuals.rolling(window=Metrics.window_index(residuals, 0.25))
        return pd.concat(
            itertools.chain([df], 
                (
                    func(rolling)
                    for func in [Metrics.variance, Metrics.ac1]
                )
            ), axis=1)
    
    @staticmethod
    def add_indices(df: pd.DataFrame, variable: str, forced: bool | np.bool, tsid: int | np.integer) -> pd.DataFrame:
        df["variable"] = np.array([variable] * len(df))
        df["forced"] = np.array([forced] * len(df))
        df["tsid"] = np.array([tsid] * len(df))
        return df.set_index(["variable", "forced", "tsid"], append=True).reorder_levels(__class__.INDICES)
        

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
            spacing=10,
            age = None,
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
        if verbose and len(remaining) != 0:
            print("Missing simulations", remaining)
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
            series = series[series.index < tcrit]
            
        return self.add_indices(pd.concat([
                    series.to_frame(name="state"), 
                    compute_residuals(series)
                    ], axis=1), str(series.name), forced, tsid)


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
    
    def __init__(self: Self, dataset: Dataset, bandwidth: int | float, tcrit: Iterable[float] = []):
        super().__init__(name=dataset.name, spacing=max(1, abs(int(dataset.df.index[-1] - dataset.df.index[0])) // 250), age=dataset.age_format)
        self.bandwidth = float(bandwidth)
        self.df = dataset.df.rename(dataset.feature_cols, axis=1)
        self.tcrit = set(t for t in tcrit if 0 <= t <= self.df.index.max())
        if len(self.tcrit) == 0:
            raise ValueError("A tipping point must be provided to test dataset", dataset.name)
        
    def compute(self: Self, series: pd.Series) -> pd.DataFrame:
        residuals = compute_residuals(series, span=self.bandwidth, type="Gaussian")
        return self.metrics(pd.concat([series.to_frame("state"), residuals], axis = 1))
    
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> dict[str, pd.DataFrame]:
        
        def per_transition(tcrit: float) -> tuple[str, pd.DataFrame]:
            
            COLUMNS = __class__.COLUMNS + ["variance", "ac1"]
            NAME = f"tcrit{np.round(tcrit, decimals=2)}"
            FILE = NAME + "_sims.csv"

            df_load = self.load(path, FILE, columns = COLUMNS)
            if df_load is not None:
                return NAME, df_load
                     
            df = interpolate(self.df[self.df.index >= tcrit]).reset_index(names="time")
            df["time"] = -df["time"]
            df = df.iloc[::-1].set_index("time")
            df = pd.concat(self.add_indices(self.compute(df[col]), str(col), True, 0) for col in df.columns)

            # Generate inputs for null simulation based on beginning of time-series
            def map_null(t: tuple[Any, pd.DataFrame]):
                col, group = t
                residuals = group["residuals"].dropna()
                window = residuals[:len(residuals) // 5]
            
                alpha = window.autocorr(lag=1)
                sigma: float = np.sqrt(window.var() * (1.0 - alpha**2.0))
                x0: float = residuals.iloc[0]  # initial condition   
                time = group.index.get_level_values("time")
                
                return str(col), alpha, sigma, x0, time
            
            # Generate a null simulation DataFrame given inputs
            def compute_null(variable: str, alpha: float, sigma: float, x0: float, tsid: int, time: pd.Index):
                length = len(time)
                dW = np.random.normal(size=length - 1)
                s = np.empty(length)
                s[:] = np.nan
                s[0] = x0

                # Run recursion in time
                for i in range(1, length):
                    # Generate noise increment N(0,1)
                    s[i] = alpha * s[i-1] + sigma * dW[i - 1]
                    
                return self.add_indices(self.compute(pd.Series(s, index=time)), variable, False, tsid)

            # Generate N null simulations for each variable
            df = pd.concat(
                iter_progress(itertools.chain(
                    [df], 
                    parallel(
                        delayed(compute_null)(variable, alpha, sigma, x0, tsid, time) 
                        for (variable, alpha, sigma, x0, time), tsid in 
                        itertools.product(
                            map(map_null, iter(df.groupby("variable"))), 
                            range(self.sims)
                    ))), verbose, self.name + " simulations"))
            
            self.save(path, FILE, df)
            
            return NAME, df
                        
        return dict(per_transition(tcrit) for tcrit in self.tcrit)
            
    
def roc(
    model: TestModel,
    kind: str,
    df: pd.DataFrame, 
    lstm: LSTM | None = None,
    path: str | None = None,
    bifurcation: Literal["All", "Hopf", "Fold", "Branch"] = "All",
) -> pd.DataFrame:
    
    LEVELS = ["i", "name", "variable", "start", "end"]
    
    list_df = []
    for variable, df in df.groupby("variable"):
        
        FILE = f"roc_{variable}_{kind}.csv"
                
        load = model.load(path, FILE, indices=LEVELS, columns=["fpr", "tpr", "thresholds", "auc"])
        if load is not None:
            list_df.append(load)
            continue
        
        df = df.reorder_levels(TestModel.INDICES)

        def roc_interval(df: pd.DataFrame, interval: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]) -> pd.DataFrame:
                
            def per_var(series: pd.Series, name: str) -> pd.DataFrame:
                series = series.dropna()

                time = series.index.get_level_values("time")
                t_start = time[0]
                t_end = series.loc[series.index.get_level_values("forced").to_numpy(dtype=bool)].reset_index().groupby("tsid")["time"].max().min() or time[-1]

                # Get prediction interval in time
                t_pred = t_start + (t_end - t_start) * interval

                # Get data within prediction interval
                trans_series: pd.Series = series[((time >= t_pred[0]) & (time <= t_pred[1]))]
            
                # Compute ROC curve and threhsolds using sklearn
                fpr, tpr, thresholds = metrics.roc_curve(trans_series.index.get_level_values("forced").astype(int), trans_series.to_numpy())

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
                
            preds = pd.concat(per_var(df[col], name) for col, name in ROC_TYPES.items() if col in df)
                
            if lstm is not None and set(LSTM.COLUMNS).issubset(df.columns):
                preds = pd.concat([preds, per_var(df[LSTM.COLUMNS[:-1]].dropna().sum(axis=1) if bifurcation == "All" else df[bifurcation].dropna(), lstm.name)])
                
            preds["start"] = interval[0]
            preds["end"] = interval[1]

            return preds.set_index(["start", "end"], append=True)
        
        df = pd.concat(map(lambda interval: roc_interval(df, interval), [np.array([0.6, 0.8]), np.array([0.8, 1])]))
        df = df.set_index(pd.Series([variable] * len(df), name = "variable"), append = True).reorder_levels(LEVELS)
    
        model.save(path, FILE, df)
        
        list_df.append(df)
    
    return pd.concat(list_df)

def counts(df: pd.DataFrame) -> pd.DataFrame:
    if set(LSTM.COLUMNS).issubset(df.columns):
        counts = df.loc[(df[LSTM.COLUMNS].notna().all(axis=1) & df.index.get_level_values("forced").to_numpy()), LSTM.COLUMNS].idxmax(axis=1).value_counts()
        return pd.DataFrame({col.lower():[counts.get(col, 0)] for col in LSTM.COLUMNS})
    else:
        return pd.DataFrame({col.lower():0 for col in LSTM.COLUMNS})

type _ModelGetter = Tuple[str, list[ModelData] | Callable[[], list[ModelData]]]

def test_models(lengths: set[int]) -> list[TestModel]:
    return [MayFold(lengths), CrHopf(lengths), CrTrans(lengths)]
        
def get_metrics(model: TestModel, lstm: LSTM | None = None, path: str | None = None, verbose: bool = True) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    
    parallel = Parallel(return_as="generator_unordered", backend="threading", n_jobs=4)
    
    def per_sim(kind: str, variable: str, forced: np.bool, tsid, df: pd.DataFrame) -> pd.DataFrame:
        assert pd.api.types.is_integer(tsid)

        FILE = f"metrics_{variable}_{"forced" if forced else "null"}_{kind}_{tsid}.csv"
        
        metrics = model.load(path, FILE, columns=["ktau_variance","ktau_ac1"] + LSTM.COLUMNS if lstm is not None else [])
        
        if metrics is None:
        
            indices = space_indices(df["variance"], spacing=model.spacing)
            
            metrics = pd.concat((Metrics.ktau(df[col], indices=indices).to_frame() for col in ["variance", "ac1"]), axis=1)

            if lstm is not None:
                metrics = pd.concat([metrics, lstm.run_on_series(df["residuals"], indices=indices, parallel=parallel)], axis=1)

            metrics = TestModel.add_indices(metrics, variable, forced, tsid).sort_index()
            
            model.save(path, FILE, metrics)
        
        return metrics
            
    metrics_dict = {
        kind:pd.concat(
            iter_progress((per_sim
                (kind, str(variable), np.bool(forced), tsid, df) 
                for (variable, forced, tsid), df 
                in full_df.groupby(TestModel.INDICES[:-1], dropna=False)
            ), verbose=verbose, desc=model.name)
        ) for kind, full_df in model.simulate(parallel, verbose=True, path=path).items()
    }
    
    return {kind:(df, roc(
                model=model,
                kind=kind,
                df=df,
                lstm=lstm,
            )) for kind, df in metrics_dict.items()}
    

def plot_metrics(model: TestModel, kind: str, metrics: pd.DataFrame, rocs: pd.DataFrame, output: str | None = None, lstm: LSTM | None = None):
    for variable, roc in rocs.groupby(["variable"], dropna=False):
            
        fig = plt.figure(figsize=(12, 6))
        
        groups = roc.groupby(["start", "end"])
        axes: Sequence[Axes] = fig.subplots(nrows=1, ncols=len(groups))
            
        for i, ((start, end), roc) in enumerate(groups):
            axis = axes[i]
            
            assert pd.api.types.is_float(start)
            assert pd.api.types.is_float(end)
            
            for name, roc in roc.groupby("name"):
                axis.plot(
                    roc["fpr"],
                    roc["tpr"],
                    label=f"{name} bif (AUC={np.round(roc["auc"].iloc[0], 2)})"
                )

                # Line y=x
                axis.plot(
                    np.linspace(0, 1, 100),
                    np.linspace(0, 1, 100),
                    color="black", 
                    linestyle="dashed",
                )

                axis.set_xlabel("False positive rate")
                axis.set_xbound(-0.01, 1)
                axis.set_ylabel("True positive rate") 
                
                axis.set_title(f"Window of {start * 100}% - {end * 100}% of data)")

        for ax in axes:
            ax.legend()
        
        fig.suptitle(f"ROC {model.name} {variable} {kind}")
        
        if output is not None:
            path = os.path.join(output, model.name)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, fig.get_suptitle() + ".png"))
        else:
            fig.show()
        
        plt.close()
    
    if lstm is not None:
        for (variable, forced), df in metrics.groupby(["variable", "forced"], dropna=False):
                
            fig = lstm.plot(
                name = model.name,
                means = lstm.calc_means(df),
                age = model.age,
                reverse = False,
            )
            
            fig.suptitle(f"{lstm.name} {model.name} {"forced" if forced else "null"} bifurcation classifcation on {variable} {kind}")
            
            if output is not None:
                path = os.path.join(output, model.name)
                os.makedirs(path, exist_ok=True)
                fig.savefig(os.path.join(path, fig.get_suptitle() + ".png"))
            else:
                fig.show()
                
            plt.close()
    
def load_and_save(models: Iterable[TestModel], lstm: LSTM | None = None, path: str | None = None, output: str | None = None):
    for model in models:
        print("Generating/Loading metrics for", model.name)
        try:
            metrics = get_metrics(lstm = lstm, path = path, model = model)
            print("Plotting metrics for", model.name)
            for kind, (metrics, rocs) in metrics.items():
                plot_metrics(model, kind=kind, metrics=metrics, rocs=rocs, output=output, lstm=lstm)
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
        path=args.metrics, 
        output=args.output, 
        lstm=LSTMLoader(args.lstm).with_args(verbose=False) if args.lstm is not None else None,
        models=test_models(set([1500, 500])),
    )