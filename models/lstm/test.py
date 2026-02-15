from typing import Any, Callable, Iterable, Sequence
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt

from models.metrics import Metrics

from ..lstm import LSTMLoader, Dataset, LSTM, StochSim, XY, DeFunc

def simulate(
    de_fun: DeFunc,
    bl: float, 
    bh: float, 
    bcrit: float, 
    ts_len: int,
    init: XY,
    sigma: XY | float,
    sims: int,
    name: str = ""
) -> tuple[list[Dataset], Any]:
    assert sims > 0
    
    model = StochSim(
        binit=bl,
        bcrit=bh,
        ts_len=ts_len,
    )
    
    def sim_tsid(tsid: int) -> pd.Series:
        df: pd.DataFrame = model.simulate(
            de_fun=de_fun,
            s0=init,
            sigma=sigma,
            clip=lambda xy: np.clip(xy, a_min=0, a_max=None)
        )
        
        # Use only x ()
        df = df[["p0"]]
        
        return Dataset(
            name=f"{name}{"_" if len(name) != 0 else ""}{tsid}_len{ts_len}",
            df=df,
            feature_cols=Dataset._convert_feature_cols(df.columns),
            age_format="",
        )
    
    return (list(map(sim_tsid, range(1, sims + 1))), model.parameter[model.parameter > bcrit].index[1]) 

def may_fold(ts_len: int, sims: int = 10):
    
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
        de_fun=de_fun,
        bl=bl,
        bh=bh,
        bcrit=bcrit,
        init=[x0],
        sigma=0.01,
        ts_len=ts_len,
        sims=sims,
    )
    
def cr_trans(ts_len: int, sf: int = 4, sims: int = 10):
        
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
        de_fun=de_fun,
        bl=al,
        bh=ah,
        bcrit=abif,
        init=[x0, y0],
        sigma=0.01,
        ts_len=ts_len,
        sims=sims,
    )
    
def cr_hopf(ts_len: int, sf: int = 4, sims: int = 10):

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
        de_fun=de_fun,
        bl=al,
        bh=ah,
        bcrit=abif,
        init=[x0, y0],
        sigma=np.array([sigma_x, sigma_y]),
        ts_len=ts_len,
        sims=sims,
    )

def compute_roc(
    df_ews_forced: pd.DataFrame, 
    df_ews_null: pd.DataFrame,
    bool_pred_early: bool = True
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    
    # --------
    # Import EWS and ML data
    # –------------)

    # # Import EWS data
    # Use EWS data in x
    
    def ktau_df(df: pd.DataFrame):
        return df[["tsid", "time", "ktau_variance", "ktau_ac1"]].dropna()
    
    def ml_df(df: pd.DataFrame):
        return df[["tsid", "time"] + LSTM.COLUMNS].dropna()
    
    df_ktau_forced = ktau_df(df_ews_forced)
    df_ktau_null = ktau_df(df_ews_null)
        
    df_ml_forced = ml_df(df_ews_forced)
    df_ml_null = ml_df(df_ews_null)

    # Add column for truth values (1 for forced, 0 for null)
    df_ktau_forced["truth value"] = 1
    df_ktau_null["truth value"] = 0

    df_ml_forced["truth value"] = 1
    df_ml_null["truth value"] = 0


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


    # Initialise lists
    list_df_ktau_preds: list[pd.DataFrame] = []
    list_df_ml_preds: list[pd.DataFrame] = []

    # Get predictions from forced trajectories
    for tsid in df_ml_forced["tsid"].unique():

        # Get EWS data to find start and transition time
        df: pd.DataFrame = df_ews_forced[(df_ews_forced["tsid"] == tsid)]
        t_start = df["time"].iloc[0]
        t_transition = (
            df[["time", "residuals"]].dropna()["time"].iloc[-1]
        )  # where the residuals end

        # Get prediction interval in time
        t_pred_start = t_start + (t_transition - t_start) * pred_interval_rel[0]
        t_pred_end = t_start + (t_transition - t_start) * pred_interval_rel[1]

        # Get ktau data specific to this variable and tsid.
        # Get data within prediction interval
        df_ktau_forced_final = df_ktau_forced[
            (df_ktau_forced["tsid"] == tsid)
            & (df_ktau_forced["time"] >= t_pred_start)
            & (df_ktau_forced["time"] <= t_pred_end)
        ]
        df_ml_forced_final = df_ml_forced[
            (df_ml_forced["tsid"] == tsid)
            & (df_ml_forced["time"] >= t_pred_start)
            & (df_ml_forced["time"] <= t_pred_end)
        ]

        # Extract 10 evenly spaced predictions across the prediciton time interval
        # We do this so some transitions don't input more data to the ROC
        # than others.
        n_predictions = 10

        # Ktau forced trajectories
        idx = np.round(np.linspace(0, len(df_ktau_forced_final) - 1, n_predictions)).astype(
            int
        )
        list_df_ktau_preds.append(df_ktau_forced_final.iloc[idx])

        # ML forced trajectories
        idx = np.round(np.linspace(0, len(df_ml_forced_final) - 1, n_predictions)).astype(
            int
        )
        list_df_ml_preds.append(df_ml_forced_final.iloc[idx])


    # Get predictions from null trajectories
    tsid_vals = df_ml_null["tsid"].unique()
    for tsid in tsid_vals:

        # Get EWS data to find start and transition time (for forced data)
        df = df_ews_forced[(df_ews_forced["tsid"] == tsid)]
        t_start = df["time"].iloc[0]
        t_transition = (
            df[["time", "residuals"]].dropna()["time"].iloc[-1]
        )  # where the residuals end

        # Get prediction interval in time
        t_pred_start = t_start + (t_transition - t_start) * pred_interval_rel[0]
        t_pred_end = t_start + (t_transition - t_start) * pred_interval_rel[1]

        # Get ktau data specific to this variable and tsid.
        # Get data within prediction interval
        df_ktau_null_final = df_ktau_null[
            (df_ktau_null["tsid"] == tsid)
            & (df_ktau_null["time"] >= t_pred_start)
            & (df_ktau_null["time"] <= t_pred_end)
        ]
        df_ml_null_final = df_ml_null[
            (df_ml_null["tsid"] == tsid)
            & (df_ml_null["time"] >= t_pred_start)
            & (df_ml_null["time"] <= t_pred_end)
        ]

        # Extract 10 evenly spaced predictions across the prediciton time interval
        # We do this so some transitions don't input more data to the ROC
        # than others.
        n_predictions = 10

        # Ktau forced trajectories
        idx = np.round(np.linspace(0, len(df_ktau_null_final) - 1, n_predictions)).astype(
            int
        )
        list_df_ktau_preds.append(df_ktau_null_final.iloc[idx])

        # ML forced trajectories
        idx = np.round(np.linspace(0, len(df_ml_null_final) - 1, n_predictions)).astype(int)
        list_df_ml_preds.append(df_ml_null_final.iloc[idx])


    # Concatenate data
    df_ktau_preds = pd.concat(list_df_ktau_preds)
    df_ml_preds = pd.concat(list_df_ml_preds)
    df_ml_preds["bif_prob"] = np.sum(df_ml_preds[LSTM.COLUMNS[:-1]].to_numpy(), axis=1)


    # -------------------
    # Get data on ML favoured bifurcation for each forced trajectory
    # -------------------

    # For each prediction, select the bifurcation that the ML gives greatest weight to
    df_ml_preds["fav_bif"] = df_ml_preds[LSTM.COLUMNS].idxmax(axis=1)

    # Count each bifurcation choice for forced trajectories
    counts = df_ml_preds[df_ml_preds["truth value"] == 1]["fav_bif"].value_counts()

    df_counts = pd.DataFrame({col.lower():[counts.get(col, 0)] for col in LSTM.COLUMNS})

    # --------------------
    # Functions to compute ROC
    # –--------------------

    # Function to compute ROC data from truth and indicator vals
    # and return a df.
    def roc_compute(df: pd.DataFrame, indicator: str):

        # Compute ROC curve and threhsolds using sklearn
        fpr, tpr, thresholds = metrics.roc_curve(df["truth value"], df[indicator])

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
        "ML": roc_compute(df_ml_preds, "bif_prob"),
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

type _Models = Iterable[tuple[str, int, tuple[list[Dataset], Any]]]

def generate_test_models():
    for ts_len in (500, 1500):
        for test in (may_fold, cr_hopf, cr_trans):
            print("Simulating", test.__name__, "for length", ts_len)
            datasets, tcrit = test(ts_len=ts_len)
            yield test.__name__, datasets, tcrit
    
def generate_metrics(lstm: LSTM, models: list[Dataset], tcrit: Any) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    
    list_df_forced: list[pd.DataFrame] = []
    list_df_null: list[pd.DataFrame] = []
    
    metrics = Metrics(window=0.25, ktau_distance=5)
        
    def run_metrics(tsid: int, tcrit) -> pd.DataFrame: 
        list_df: list[pd.DataFrame] = []
        for var in dataset.feature_cols.keys():
            df = metrics.run_on_series(dataset.df[var], transition=tcrit)
            df = df.join(lstm.calc_means(lstm.run_on_series(df["residuals"])))
            df.insert(0, "tsid", tsid + 1)
            list_df.append(df.reset_index())
        return pd.concat(list_df)
        
    ts_len = None
    for tsid, dataset in enumerate(models):
        
        if ts_len is None:
            ts_len = len(dataset.df)
        else:
            assert ts_len == len(dataset.df)
        
        list_df_forced.append(run_metrics(tsid, None))
        list_df_null.append(run_metrics(tsid, tcrit))
    
    return ts_len, pd.concat(list_df_forced), pd.concat(list_df_null)

def test(lstm: LSTM, models: Iterable[tuple[str, list[Dataset], Any]], output: str | None = None):
    
    for name, datasets, tcrit in models:
        ts_len, forced, null = generate_metrics(lstm, datasets, tcrit)
        
        for variable in null["variable"].unique():
            for type in ["early", "late"]:
                
                df_forced = forced[forced["variable"] == variable]
                df_null = null[null["variable"] == variable]
            
                rocs, _ = compute_roc(
                    df_forced,
                    df_null,
                    bool_pred_early= type == "early",
                )
                
                fig = plot_roc(
                    rocs,
                    name + " " + variable + " " + type,
                    ts_len
                )
                
                if output:
                    fig.savefig(os.path.join(output, fig.get_suptitle() + ".png"))
        
        
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Tester',
                    description='Tests trained models against simulated data')
    parser.add_argument('modelpath', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--extension', '-e', type=str)
    args = parser.parse_args()
    
    lstm = LSTMLoader(args.modelpath, extension=args.extension).with_args(verbose=False)
    
    test(lstm, models=generate_test_models(), output=args.output)
    

        