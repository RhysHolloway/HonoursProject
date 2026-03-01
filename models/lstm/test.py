import functools
from typing import Any, Callable, Final, Hashable, Iterable, Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import os.path

from models.metrics import Metrics

from ..lstm import LSTMLoader, Dataset, LSTM, StochSim, Array, DeFunc

def simulate(
    lengths: Sequence[int],
    de_fun: DeFunc,
    bl: float, 
    bh: float, 
    bcrit: float,
    init: Array,
    sigma: Array | float,
    sims: int,
    name: str = ""
) -> list[tuple[list[Dataset], float]]:
    assert sims > 0
    
    max_len = max(lengths)
    model = StochSim(
        binit=bl,
        bcrit=bh,
        ts_len=max_len,
    )
    
    def sim_tsid(tsid: int) -> Dataset:
        df: pd.DataFrame = model.simulate(
            de_fun=de_fun,
            s0=init,
            sigma=sigma,
            clip=lambda xy: np.clip(xy, a_min=0, a_max=None)
        )
        
        # Use only x ()
        df = df[["p0"]]
        df.insert(0,"tsid", tsid)
        
        return Dataset(
            name=f"{name}{"_" if len(name) != 0 else ""}{tsid}_len{len(df)}",
            df=df,
            feature_cols=Dataset._convert_feature_cols(df.columns.drop("tsid")),
            age_format="",
        )
        
    maximums = (list(map(sim_tsid, range(1, sims + 1))), model.parameter[model.parameter > bcrit].index[1])
    
    def of_length(ts_len: int):
        return [
            Dataset(
                name=f"{name}{"_" if len(name) != 0 else ""}{tsid+1}_len{ts_len}",
                df=dataset.df.iloc[-ts_len:],
                feature_cols=dataset.feature_cols,
                age_format="",
            )
        for tsid, dataset in enumerate(maximums[0])], maximums[1]
    
    sims: list[tuple[list[Dataset], float]] = []
    
    for ts_len in lengths:
        if ts_len != max_len:
            sims.append(of_length(ts_len))
    
    sims.append(maximums)
    return sims 

def may_fold(
    lengths: Sequence[int],
    sims: int = 10
):
    
    r = 1  # growth rate
    k = 1  # carrying capacity
    s = 0.1  # half-saturation constant of harvesting function
    bl = 0.15  # bifurcation parameter low
    bh = 0.27  # bifurcation parameter high
    bcrit = 0.260437  # bifurcation point (computed in Mathematica)
    x0 = 0.8197  # intial condition (equilibrium value computed in Mathematica)
        
    def de_fun(x: np.ndarray[float], a: float) -> np.ndarray[float]:
        x = x[0]
        return np.array([
            r * x * (1 - x / k) - a * (x**2 / (s**2 + x**2))
        ])

    return simulate(
        lengths=lengths,
        de_fun=de_fun,
        bl=bl,
        bh=bh,
        bcrit=bcrit,
        init=[x0],
        sigma=0.01,
        sims=sims,
    )
    
def cr_trans(
    lengths: Sequence[int],sf: int = 4, sims: int = 10):
        
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
    
    def de_fun(s: np.ndarray[float], a: float) -> np.ndarray[float]:
        x, y = s
        return np.array([
            r * x * (1 - x / k) - (a * x * y) / (1 + a * h * x),
            e * a * x * y / (1 + a * h * x) - m * y
        ])
    
    return simulate( 
                    lengths=lengths,       
        de_fun=de_fun,
        bl=al,
        bh=ah,
        bcrit=abif,
        init=[x0, y0],
        sigma=0.01,
        sims=sims,
    )
    
def cr_hopf(
    lengths: Sequence[int],sf: int = 4, sims: int = 10):

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
    
    def de_fun(s: np.ndarray[float], a: float) -> np.ndarray[float]:
        x, y = s
        return np.array([
            r * x * (1 - x / k) - (a * x * y) / (1 + a * h * x),
            e * a * x * y / (1 + a * h * x) - m * y
        ])
        
    return simulate(
        lengths=lengths, 
        de_fun=de_fun,
        bl=al,
        bh=ah,
        bcrit=abif,
        init=[x0, y0],
        sigma=np.array([sigma_x, sigma_y]),
        sims=sims,
    )

def compute_roc(
    df_ews_forced: pd.DataFrame, 
    df_ews_null: pd.DataFrame,
    bool_pred_early: bool = True,
    model_name: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    
    # --------
    # Import EWS and ML data
    # –------------

    # # Import EWS data
    # Use EWS data in x
    
    def ktau_df(df: pd.DataFrame):
        return df[["tsid", "time", "ktau_variance", "ktau_ac1"]].dropna()
    
    def ml_df(df: pd.DataFrame):
        return df[["tsid", "time"] + LSTM.COLUMNS].dropna()

    # ---------------------------
    # Get predictions from trajectories
    # --------------------------

    # Time interval relative to transition point for where to make predictions
    # as proportion of dataset
    if bool_pred_early:
        pred_interval_rel = np.array([0.6, 0.8])
    else:
        # Late interval for predictions
        pred_interval_rel = np.array([0.8, 1])

    def predictions(df: pd.DataFrame, truth: int):
        t_start = df["time"].iloc[0]
        t_transition = df[["time", "residuals"]].dropna()["time"].iloc[-1] # where the residuals end

        # Get prediction interval in time
        t_pred_start = t_start + (t_transition - t_start) * pred_interval_rel[0]
        t_pred_end = t_start + (t_transition - t_start) * pred_interval_rel[1]

        # Get data within prediction interval
        df_timed = df[(df["time"] >= t_pred_start) & (df["time"] <= t_pred_end)]

        # Extract 10 evenly spaced predictions across the prediciton time interval
        # We do this so some transitions don't input more data to the ROC
        # than others.
        n_predictions = 10
        
        def extract(df: pd.DataFrame):
            df = df.iloc[np.round(np.linspace(0, len(df) - 1, n_predictions)).astype(int)]
            df["truth_value"] = truth
            return df

        return (
            extract(ktau_df(df_timed)), 
            extract(ml_df(df_timed))
        )
        
    forced_ktau, forced_ml = zip(*(predictions(df, 1) for _, df in df_ews_forced.groupby("tsid")))
    null_ktau, null_ml = zip(*(predictions(df, 0) for _, df in df_ews_null.groupby("tsid")))

    # Concatenate data
    df_ktau_preds: pd.DataFrame = pd.concat(forced_ktau + null_ktau)
    df_ml_preds: pd.DataFrame = pd.concat(forced_ml + null_ml)
    
    df_ml_preds["bif_prob"] = np.sum(df_ml_preds[LSTM.COLUMNS[:-1]].to_numpy(), axis=1)


    # -------------------
    # Get data on ML favoured bifurcation for each forced trajectory
    # -------------------

    # For each prediction, select the bifurcation that the ML gives greatest weight to
    df_ml_preds["fav_bif"] = df_ml_preds[LSTM.COLUMNS].idxmax(axis=1)

    # Count each bifurcation choice for forced trajectories
    counts = df_ml_preds[df_ml_preds["truth_value"] == 1]["fav_bif"].value_counts()

    df_counts = pd.DataFrame({col.lower():[counts.get(col, 0)] for col in LSTM.COLUMNS})

    # --------------------
    # Functions to compute ROC
    # –--------------------

    # Function to compute ROC data from truth and indicator vals
    # and return a df.
    def roc_compute(df: pd.DataFrame, indicator: str):

        # Compute ROC curve and threhsolds using sklearn
        fpr, tpr, thresholds = metrics.roc_curve(df["truth_value"], df[indicator])

        return pd.DataFrame({
            "fpr": fpr, 
            "tpr": tpr, 
            "thresholds": thresholds, 
            "auc": metrics.auc(fpr, tpr) # Compute AUC (area under curve)
        })


    # ---------------------
    ## Compute ROC data
    # –--------------------
    
    return {
        model_name or "DL Model": roc_compute(df_ml_preds, "bif_prob"),
        "Variance": roc_compute(df_ktau_preds, "ktau_variance"),
        "Lag-1 AC": roc_compute(df_ktau_preds, "ktau_ac1"),
    }, df_counts
    
def plot_roc(rocs: dict[str, pd.DataFrame], name: str, ts_len: int):

    fig = plt.figure()
    axes: Axes = fig.subplots()
        
    for roc_name, df_roc in rocs.items():
        axes.plot(
            df_roc["fpr"],
            df_roc["tpr"],
            label=f"{roc_name} bif (AUC={np.round(df_roc["auc"].iloc[0], 2)})"
        )

    # Line y=x
    axes.plot(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, 100),
        color="black", 
        linestyle="dashed",
    )

    axes.set_xlabel("False positive rate")
    axes.set_xbound(-0.01, 1)
    axes.set_ylabel("True positive rate")

    fig.legend()
    fig.suptitle(f"ROC {name} len{ts_len}")
    
    return fig

type _Model = tuple[str, list[Dataset], float]

# def generate_test_models(lengths: Sequence[int] = [1500, 500]):
#         for test in (may_fold, cr_hopf, cr_trans):
#             print("Generating", test.__name__, "of lengths", lengths)
#             for datasets, tcrit in test(lengths=lengths):
#                     yield test.__name__, datasets, tcrit

def save_test_models(folder: str, models: Iterable[_Model]):
    import os.path
    tcrits: dict[int, pd.Series] = dict()
    os.makedirs(folder, exist_ok=True)
    for model in models:
        name, datasets, tcrit = model
        ts_len = len(datasets[0].df)
        len_folder = os.path.join(folder, f"len{ts_len}")
        os.makedirs(len_folder, exist_ok=True)
        
        pd.concat([dataset.df.reset_index()[["tsid", "time"] + list(dataset.feature_cols.keys())] for dataset in datasets]).to_csv(os.path.join(len_folder, name + ".csv"), index=False)
        
        if ts_len not in tcrits:
            tcrits[ts_len] = pd.read_csv(os.path.join(len_folder, "transitions.csv"), names=["model", "tcrit"])["tcrit"]
        
        tcrits[ts_len][name] = tcrit
    
    for ts_len, series in tcrits.items():
        series.to_csv(os.path.join(folder, f"len{ts_len}", "transitions.csv"))
        
            
def read_test_models(folder: str, name: Hashable | None = None):
    import os.path
    for len_file in os.listdir(folder):
        len_folder = os.path.join(folder, len_file)
        if os.path.isdir(len_folder) and len_file.startswith("len") and len_file[3:].isdigit():
            ts_len = int(len_file[3:])
            transitions = pd.read_csv(os.path.join(len_folder, "transitions.csv"), index_col=0, names=["model", "tcrit"])["tcrit"]
            for mname, tcrit in (transitions.items() if name is None else [(name, transitions[name])]):
                model = pd.read_csv(os.path.join(len_folder, mname + ".csv"), index_col=False)
                yield mname, [Dataset(
                    name=f"{mname}_{tsid}_len{ts_len}",
                    df=df.set_index("time"),
                    feature_cols=Dataset._convert_feature_cols(df.columns.drop(["tsid", "time"])),
                    age_format="",
                ) for tsid, df in model.groupby("tsid", as_index=False)], tcrit
            
type _Metrics = tuple[pd.DataFrame, pd.DataFrame]
def generate_metrics(name: str, lstm: LSTM, models: list[Dataset], tcrit: float, output: str | None = None) -> _Metrics:
    
    list_df_forced: list[pd.DataFrame] = []
    list_df_null: list[pd.DataFrame] = []
    
    metrics = Metrics(window=0.25, ktau_distance=5)
        
    def run_metrics(tsid: int, tcrit: float | None) -> pd.DataFrame: 
        list_df: list[pd.DataFrame] = []
        for var in dataset.feature_cols.keys():
            df = metrics.run_on_series(dataset.df[var], transition=tcrit)
            df = df.join(lstm.calc_means(lstm.run_on_series(series=df["residuals"])))
            df.insert(0, "tsid", tsid + 1)
            list_df.append(df.reset_index())
        return pd.concat(list_df).rename(dataset.feature_cols)
        
    ts_len = None
    for tsid, dataset in enumerate(models):
        
        if ts_len is None:
            ts_len = len(dataset.df)
        else:
            assert ts_len == len(dataset.df)
        
        list_df_forced.append(run_metrics(tsid, None))
        list_df_null.append(run_metrics(tsid, tcrit))
        
    forced, null = pd.concat(list_df_forced), pd.concat(list_df_null)
        
    if output is not None:
        import os.path
        forced.to_csv(os.path.join(output, f"{name}_len{ts_len}_ews_roc_forced.csv"), index=False) 
        null.to_csv(os.path.join(output, f"{name}_len{ts_len}_ews_roc_null.csv"), index=False)
    
    return forced, null
        
def load_metric(folder: str, name: str, ts_len: int) -> _Metrics | None:
    forced = os.path.join(folder, f"{name}_len{ts_len}_ews_roc_forced.csv")
    null = os.path.join(folder, f"{name}_len{ts_len}_ews_roc_null.csv")
    if os.path.exists(forced) and os.path.exists(null):
        return pd.read_csv(forced, index_col=False), pd.read_csv(null, index_col=False)
    else:
        return None
        
def load_metrics(folder: str, ts_len: int) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:    
    return list((name, ) + load_metric(folder, name, ts_len) for name in set(name[:name.index(f"_len{ts_len}_ews_roc")] for name in os.listdir(folder) if "_ews_roc_" in name and name.endswith(".csv")))

def plot_metrics(name: str, forced: pd.DataFrame, null: pd.DataFrame, output: str | None = None, model_name: str | None = None):           
    for variable in forced["variable"].unique():
        for type in ["early", "late"]:
            
            df_forced = forced[forced["variable"] == variable]
            df_null = null[null["variable"] == variable]
        
            rocs, _ = compute_roc(
                df_forced,
                df_null,
                bool_pred_early = type == "early",
                model_name=model_name,
            )
            
            fig = plot_roc(
                rocs,
                f"{name} {variable} {type}",
                len(df_forced)
            )
            
            if output is not None:
                import os.path
                fig.savefig(os.path.join(output, fig.get_suptitle() + ".png"))

type _ModelGenerator = dict[str, Callable[[], tuple[list[Dataset], float]]]

def get_or_generate_models(lengths: Sequence[int] = [1500, 500], path: str | None = None):
    models: list[_Model] = []
    for name, mfunc in map(lambda mfunc: (mfunc.__name__, functools.partial(mfunc, lengths=lengths)), [may_fold, cr_hopf, cr_trans]):
        if path is not None:
            loaded = list(read_test_models(path, name))
            if len(loaded) != 0:
                print("Loaded", name)
                models += loaded
                continue
    
        for datasets, tcrit in mfunc():
            print("Generating", name)
            model = (name, datasets, tcrit)
            models.append(model)
            if path is not None:
                save_test_models(path, [model])

    return models
    
def load_and_save(metrics_path: str | None, output: str | None, lstm: LSTM | None, models: Iterable[_Model]):
    for name, datasets, tcrit in models:
        ts_len = len(datasets[0].df)
        metric = load_metric(metrics_path, name, ts_len) if metrics_path is not None else None
        if metric is None:
            if lstm is None:
                raise ValueError("Please provide an LSTM to generate missing metrics!")
            print("Generating metrics for", name)
            metric = generate_metrics(name=name, lstm=lstm, models=datasets, tcrit=tcrit, output=metrics_path)
        forced, null = metric
        plot_metrics(name, forced, null, output=output, model_name=lstm.name if lstm is not None else None)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Tester',
                    description='Tests trained models against simulated data')
    parser.add_argument('output', type=str)
    parser.add_argument('--lstm', '-l', type=str)
    parser.add_argument('--simulations', '-s', type=str|bool)
    parser.add_argument('--metrics', '-i', type=str)
    args = parser.parse_args()
    
    try:
        load_and_save(
            metrics_path=args.metrics, 
            output=args.output, 
            lstm=LSTMLoader(args.lstm).with_args(verbose=False) if args.lstm is not None else None,
            models=get_or_generate_models(path=args.simulations),
        )
    except ValueError as e:
        raise ValueError("Please provide either -m for an LSTM model path or -i for an input path of pre-generated metrics!", e)