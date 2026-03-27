import abc
import itertools
from typing import Any, Callable, Final, Iterable, Literal, Self, Sequence, Tuple
from joblib import Parallel, delayed
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os.path

from models import Dataset, compute_residuals, interpolate, iter_progress
from models.metrics import Metrics
from models.lstm import LSTMLoader, LSTM, StochSim, Array, DeFunc

type ModelData = tuple[pd.Series, pd.Series, float]

class TestModel(metaclass = abc.ABCMeta):
        
    INDICES: Final[list[str]] = ["variable", "forced", "tsid", "time"]
    
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
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> pd.Series:
        pass
    
    @abc.abstractmethod
    def for_lengths(self: Self) -> Iterable[int]:
        pass
    
    def load(self: Self, checks: Callable[[pd.Series], bool] = lambda _: True, path: str | None = None) -> tuple[pd.Series, bool]:
        try:
            if path is None:
                raise
            
            df = pd.read_csv(os.path.join(path, self.name + "_sims.csv"), index_col=__class__.INDICES)["residuals"]
            
            if not checks(df):
                raise
            
            return df, True
        except:
            return pd.DataFrame(columns=__class__.INDICES + ["residuals"]).set_index(__class__.INDICES)["residuals"], False
        
    def save(self: Self, series: pd.Series, path: str | None = None):
        if path is not None:
            series.to_csv(os.path.join(path, self.name + "_sims.csv")) 

class SimModel(TestModel):
    
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
        
    def for_lengths(self: Self) -> Iterable[int]:
        return self.lengths
        
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> pd.Series:
        
        def checks(df: pd.Series):
            return not all(len(series) < max(self.lengths) for _, series in df.groupby(by=__class__.INDICES[:-1], dropna=False))
        
        df, success = self.load(checks, path=path)
        if success:
            return df
                
        exists = df.index.get_level_values("tsid").unique()
        required = self.sims - len(exists)
        if required > 0:
            
            def compute(tsid: int) -> pd.Series:
                forced = self._simulate_once(max(self.lengths), False, tsid)
                null = self._simulate_once(max(self.lengths), True, tsid)
                return pd.concat([forced, null])
            
            df: pd.Series = pd.concat(iter_progress(itertools.chain([df], parallel(delayed(compute)(i) for i in set(range(self.sims)).difference(exists))), verbose, desc=self.name + " simulations"))
            
            self.save(df, path=path)
                
        return df              
    
    def _simulate_once(
        self: Self,
        length: int,
        forced: bool,
        tsid: int,
    ) -> pd.Series:

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
            
        residuals = compute_residuals(series, type="Lowess") # Trained on lowess
        residuals = residuals[residuals.notna()].to_frame()
        
        residuals.insert(0, "variable", [series.name] * len(residuals))
        residuals["time"] = residuals.index
        residuals["forced"] = forced
        residuals["tsid"] = tsid
        
        return residuals.set_index(__class__.INDICES)["residuals"]


class MayFold(SimModel):
    
    @staticmethod   
    def de_fun(x: Array, a: float) -> Array:
            
        R = 1  # growth rate
        K = 1  # carrying capacity
        S = 0.1  # half-saturation constant of harvesting function
        
        x = x[0]
        return np.array([
            R * x * (1 - x / K) - a * (x**2 / (S**2 + x**2))
        ])
        
    def __init__(self: Self, lengths: set[int]):
        super().__init__(
            name="may_fold",
            de_fun=__class__.de_fun,
            bl = 0.15,  # bifurcation parameter low
            bh = 0.27,  # bifurcation parameter high
            bcrit = 0.260437,  # bifurcation point (computed in Mathematica)
            init=np.array([0.8197]),  # intial condition (equilibrium value computed in Mathematica)
            sigma=0.01,
            lengths=lengths,
        )
    
class CrTrans(SimModel):  

    def __init__(self: Self, lengths: set[int], sf: int = 4):
        
        # Model parameters
        r = 1 * sf
        k = 1.7
        h = 0.6 / sf
        e = 0.5
        m = 0.5 * sf
        
        al = 0.5 * sf  # control parameter initial value
        ah = 1.5 * sf  # control parameter final value
        abif = 1.4 * sf  # bifurcation point (computed in Mathematica)
        x0 = 1  # intial condition (equilibrium value computed in Mathematica)
        y0 = 0.412
        
        def de_fun(s: Array, a: float) -> Array:
            x, y = s
            return np.array([
                r * x * (1 - x / k) - (a * x * y) / (1 + a * h * x),
                e * a * x * y / (1 + a * h * x) - m * y
            ])
        
        super().__init__(
            name="cr_trans",
            de_fun=de_fun,
            bl=al,
            bh=ah,
            bcrit=abif,
            init=np.array([x0, y0]),
            sigma=0.01,
            lengths=lengths,
        )
    
class CrHopf(SimModel):
    def __init__(self: Self, lengths: set[int], sf: int = 4):

        # Model parameters
        # sf = 4  # scale factor
        sigma_x = 0.01  # noise intensity
        sigma_y = 0.01
        r = 1 * sf
        k = 1.7
        h = 0.6 / sf
        e = 0.5
        m = 0.5 * sf
        al = 3 * sf  # control parameter initial value
        ah = 4 * sf  # control parameter final value
        abif = 3.923 * sf  # bifurcation point (computed in Mathematica)
        x0 = 1  # intial condition (equilibrium value computed in Mathematica)
        y0 = 0.412    
        
        def de_fun(s: Array, a: float) -> Array:
            x, y = s
            return np.array([
                r * x * (1 - x / k) - (a * x * y) / (1 + a * h * x),
                e * a * x * y / (1 + a * h * x) - m * y
            ])
            
        super().__init__(
            name="cr_hopf",
            de_fun=de_fun,
            bl=al,
            bh=ah,
            bcrit=abif,
            init=np.array([x0, y0]),
            sigma=np.array([sigma_x, sigma_y]),
            lengths=lengths,
        )
        
class DatasetModel(TestModel):
    
    def __init__(self: Self, dataset: Dataset, bandwidth: int, tcrit: list[float]):
        super().__init__(name=dataset.name, spacing=max(1, abs(int(dataset.df.index[-1] - dataset.df.index[0])) // 150), age=dataset.age_format)
        self.df = dataset.df
        self.bandwidth = float(bandwidth)
        self.tcrit = set(tcrit < self.df.index.max())
    
    def for_lengths(self: Self) -> Iterable[int]:
        return [len(self.df)]
    
    def simulate(self: Self, parallel: Parallel, verbose: bool, path: str | None = None) -> pd.Series:
        
        series, success = self.load(path=path)
        if success:
            return series
        
        list_df: list[pd.Series] = []
        for col, series in interpolate(self.df).items():
            for tsid, tcrit in enumerate(itertools.chain([self.df.index.max()], self.tcrit)):
                forced_series = series[series.index <= tcrit]
                forced_series = forced_series[::-1]
                forced_series.index = -forced_series.index
                residuals = compute_residuals(forced_series, span = self.bandwidth, type = "Gaussian")
                index = residuals.index.to_frame()
                index["tsid"] = tsid
                index.insert(0, "variable", [col] * len(residuals))
                index["forced"] = True
                residuals.index = pd.MultiIndex.from_frame(index)
                list_df.append(residuals)
            
        series = pd.concat(list_df)
        
        window = series[:len(series) // 5]
        
        alpha = window.autocorr(lag=1)
        sigma = np.sqrt(window.var() * (1 - alpha**2))        
        x0 = series.iloc[0]  # initial condition   
        
        # Loop through each simulation number
        def compute(tsid: int, variable, series: pd.Series):
            dW = np.random.normal(size=len(series) - 1)
            s = np.empty(len(series))
            s[:] = np.nan
            s[0] = x0

            # Run recursion in time
            for i in range(1, len(series)):
                # Generate noise increment N(0,1)
                s[i] = alpha * s[i-1] + sigma * dW[i - 1]
                
            residuals = compute_residuals(pd.Series(s, index=series.index.get_level_values("time")), type="Gaussian", span = self.bandwidth)
                
            index = residuals.index.to_frame(index=False)
            index["tsid"] = tsid
            index.insert(0, "variable", [variable] * len(index))
            index["forced"] = False
            
            return pd.Series(residuals.to_numpy(), index=pd.MultiIndex.from_frame(index), name="residuals")
        
        def loop(series: pd.Series):
            for tsid in range(self.sims):
                for variable, series in series.groupby("variable", dropna=False):
                    yield tsid, variable, series

        series = pd.concat(iter_progress(itertools.chain([series], parallel(delayed(compute)(tsid, variable, series) for tsid, variable, series in loop(series[series.index.get_level_values("tsid") == 0]))), verbose, self.name + " simulations"))
        
        self.save(series, path=path)
        
        return series
            
    
def roc(
    df: pd.DataFrame, 
    lstm: LSTM | None = None,
    bifurcation: Literal["All", "Hopf", "Fold", "Branch"] = "All",
) -> pd.DataFrame:

    def predictions(df: pd.DataFrame, early: bool) -> pd.DataFrame:
        
        # Time interval relative to transition point for where to make predictions
        # as proportion of dataset
        INTERVAL = np.array([0.6, 0.8]) if early else np.array([0.8, 1])
            
        def per_var(series: pd.Series, name: str):
            series = series.dropna().to_frame().reorder_levels(TestModel.INDICES)[series.name]
            
            transitions = series[series.index.get_level_values("forced").to_numpy()].reset_index().groupby("tsid")["time"].max()
            
            list_df = []
            
            for transition in transitions.unique():
                forced_tsids: pd.Index = transitions[transitions == transition].index
                forced_series = series.loc[pd.IndexSlice[:, True, forced_tsids, :]]
                null_series = series[~series.index.get_level_values("forced").to_numpy()].groupby("tsid").filter(lambda s: s.index.get_level_values("time").max() >= transition)
                trans_series = pd.concat([forced_series, null_series])
                time = trans_series.index.get_level_values("time")
                t_start = time[0]

                # Get prediction interval in time
                t_pred = t_start + (transition - t_start) * INTERVAL

                # Get data within prediction interval
                trans_series: pd.Series = trans_series[((time >= t_pred[0]) & (time <= t_pred[1]))]
            
                # Compute ROC curve and threhsolds using sklearn
                fpr, tpr, thresholds = metrics.roc_curve(trans_series.index.get_level_values("forced").astype(int), trans_series.to_numpy())

                df = pd.DataFrame({
                    "fpr": fpr, 
                    "tpr": tpr, 
                    "thresholds": thresholds, 
                    "auc": metrics.auc(fpr, tpr), # Compute AUC (area under curve)
                    "name": name,
                    "transition": transition,
                })
                
                df.index.name = "i"
                list_df.append(df.set_index(["name", "transition"], append=True))
                
            return pd.concat(list_df)
        
        ROC_TYPES: dict[str, str] = {
            "ktau_variance": "Variance",
            "ktau_ac1": "Lag-1 AC",
        }
            
        preds = pd.concat(per_var(df[col], name) for col, name in ROC_TYPES.items() if col in df)
            
        if lstm is not None and set(LSTM.COLUMNS).issubset(df.columns):
            preds = pd.concat([preds, per_var(df[LSTM.COLUMNS[:-1]].dropna().sum(axis=1) if bifurcation == "All" else df[bifurcation].dropna(), lstm.name)])

        return preds.set_index(pd.Series([early] * len(preds), name = "early"), append=True)
    
    return pd.concat([predictions(df, early=True), predictions(df, early=False)])

def counts(df: pd.DataFrame) -> pd.DataFrame:
    if set(LSTM.COLUMNS).issubset(df.columns):
        counts = df.loc[(df[LSTM.COLUMNS].notna().all(axis=1) & df.index.get_level_values("forced").to_numpy()), LSTM.COLUMNS].idxmax(axis=1).value_counts()
        return pd.DataFrame({col.lower():[counts.get(col, 0)] for col in LSTM.COLUMNS})
    else:
        return pd.DataFrame({col.lower():0 for col in LSTM.COLUMNS})

type _ModelGetter = Tuple[str, list[ModelData] | Callable[[], list[ModelData]]]

def test_models(lengths: set[int]) -> list[TestModel]:
    return [MayFold(lengths), CrHopf(lengths), CrTrans(lengths)]
        
def get_metrics(model: TestModel, lstm: LSTM | None = None, path: str | None = None) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    
    verbose = True
    
    parallel = Parallel(return_as="generator_unordered", backend="threading", n_jobs=4)
    
    def get_time_series():
        for (variable, forced, tsid), residuals in model.simulate(parallel, verbose=True, path=path).groupby(TestModel.INDICES[:-1], dropna=False):
            assert pd.api.types.is_bool(forced)
            assert pd.api.types.is_integer(tsid)
            residuals.index = residuals.index.get_level_values("time")
            residuals.name = model.name
            for length in model.for_lengths():
                yield variable, forced, tsid, length, residuals
    
    list_df: dict[int, list[pd.DataFrame]] = dict()
    for variable, forced, tsid, length, residuals in iter_progress(get_time_series(), verbose=verbose, desc=model.name):
        if length not in list_df:
            list_df[length] = []
            
        if path is not None:
            file_path = os.path.join(path, f"{model.name}_{variable}_{"forced" if forced else "null"}_len{length}_metrics_{tsid}.csv") # type: ignore
            try:
                list_df[length].append(pd.read_csv(file_path, index_col=TestModel.INDICES))
                continue
            except:
                pass
        
        len_residuals = residuals.iloc[-min(length, len(residuals)):]
        
        rolling = len_residuals.rolling(window=Metrics.window_index(len_residuals, 0.25))
        
        metrics: pd.DataFrame = pd.concat((Metrics.ktau(func(rolling), spacing=model.spacing).to_frame() for func in [Metrics.variance, Metrics.ac1]), axis=1)

        if lstm is not None:
            metrics = pd.concat([metrics, lstm.run_on_series(len_residuals, spacing=model.spacing, parallel=parallel)], axis=1)
            
        index: pd.DataFrame = metrics.index.to_frame(index=False)
        index["variable"] = np.array([variable] * len(index))
        index["forced"] = np.array([forced] * len(index))
        index["tsid"] = np.array([tsid] * len(index))
        
        metrics = metrics.set_index(pd.MultiIndex.from_frame(index)).reorder_levels(TestModel.INDICES).sort_index()
        
        if path is not None:
            os.makedirs(os.path.dirname(file_path), exist_ok=True) # type: ignore
            metrics.to_csv(file_path) # type: ignore
        
        list_df[length].append(metrics)
            
    metrics_dict = {length:pd.concat(list_df) for length, list_df in list_df.items()}
    
    def compute_roc(length: int, df: pd.DataFrame):
        list_df = []
        for variable, df_var in df.groupby("variable"):
            
            LEVELS = ["i", "name", "transition", "variable", "early"]
            
            if path is not None:
                roc_path = os.path.join(path, f"{model.name}_{variable}_len{length}_roc.csv")
                os.makedirs(os.path.dirname(path), exist_ok=True) # type: ignore
                try:
                    list_df.append(pd.read_csv(roc_path, index_col=LEVELS))
                    continue
                except:
                    pass
                
            roc_df = roc(
                df_var,
                lstm,
            )
            
            roc_df = roc_df.set_index(pd.Series([variable] * len(roc_df), name = "variable"), append = True).reorder_levels(LEVELS)
            
            if path is not None:
                roc_df.to_csv(roc_path) # type: ignore
                
            list_df.append(roc_df)
        return pd.concat(list_df)
    
    return {length:(df, compute_roc(length, df)) for length, df in metrics_dict.items()}
    

def plot_metrics(model: TestModel, ts_len: int, metrics: pd.DataFrame, rocs: pd.DataFrame, output: str | None = None, lstm: LSTM | None = None):
    for (variable, transition), roc in rocs.groupby(["variable", "transition"], dropna=False):
            
        fig = plt.figure(figsize=(12, 6))
        early = roc.index.get_level_values("early").unique()
        axes: dict[Any, Axes] = {early[i]:ax for i, ax in enumerate(fig.subplots(nrows=1, ncols=len(early)))}
            
        for early, roc in rocs.groupby("early"):
            axis = axes[early]
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
                
                match early:
                    case True:
                        title = "Early (Window of 60% - 80% of data)"
                    case False:
                        title = "Late (Window of 80% - 100% of data)"
                    case _:
                        title = ""    
                
                axis.set_title(title)

        for ax in axes.values():
            ax.legend()
        
        fig.suptitle(f"ROC {model.name} {variable} at {transition} len{ts_len}")
        
        if output is not None:
            os.makedirs(output, exist_ok=True)
            fig.savefig(os.path.join(output, fig.get_suptitle() + ".png"))
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
            
            fig.suptitle(f"{lstm.name} {model.name} {"forced" if forced else "null"} bifurcation classifcation on {variable} length {ts_len}")
            
            if output is not None:
                os.makedirs(output, exist_ok=True)
                fig.savefig(os.path.join(output, fig.get_suptitle() + ".png"))
            else:
                fig.show()
                
            plt.close()
    
def load_and_save(models: Iterable[TestModel], lstm: LSTM | None = None, path: str | None = None, output: str | None = None):
    for model in models:
        print("Generating/Loading metrics for", model.name)
        metrics = get_metrics(lstm = lstm, path = path, model = model)
        print("Plotting metrics for", model.name)
        for length, (metrics, rocs) in metrics.items():
            plot_metrics(model, ts_len=length, metrics=metrics, rocs=rocs, output=output, lstm=lstm)
        
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