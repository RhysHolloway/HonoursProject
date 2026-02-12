from typing import Any, Callable, Iterable, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from ..lstm import LSTMLoader, Dataset, LSTM, StochSim, XY, DeFunc

def simulate(
    de_fun: DeFunc,
    bl: float, 
    bh: float, 
    bcrit: float, 
    ts_len: int,
    init: XY,
    sigma: XY | float,
    sims: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert sims > 0
    
    model = StochSim(
        binit=bl,
        bcrit=bh,
        ts_len=ts_len,
    )
    
    def sim_tsid(tsid: int) -> pd.DataFrame:
        df: pd.DataFrame = model.simulate(
            de_fun=de_fun,
            s0=init,
            sigma=sigma,
            clip=lambda xy: np.clip(xy, a_min=0, a_max=None)
        )
        df.insert(0, "tsid", tsid)
        return df
    
    
    df_traj_filt: pd.DataFrame = pd.concat(map(sim_tsid, range(1, sims + 1)), ignore_index=True)
    
    import ewstools
    rw = 0.25  # rolling window
    span = 0.2  # span for Lowess smoothing
    
    def ews(tcrit: pd.Series | None) -> pd.DataFrame:
    
        list_df: list[pd.DataFrame] = []

        # loop through realisation number
        for tsid, df_traj in df_traj_filt.groupby(by="tsid"):
            for var, series in df_traj.set_index("Time").drop(columns=["tsid"]).items():
                ts = ewstools.TimeSeries(series, transition=tcrit)
                ts.detrend(method="Lowess", span=span)
                ts.compute_var(rolling_window=rw)
                ts.compute_auto(rolling_window=rw, lag=1)
                df_ews_temp = ts.state.join(ts.ews).reset_index()

                # Include a column in the DataFrames for realisation number and variable
                df_ews_temp.insert(0, "tsid", tsid)
                df_ews_temp.insert(1, "Variable", var)

                # Add DataFrames to list
                list_df.append(df_ews_temp)
        return pd.concat(list_df, ignore_index=True)    
    
    df_ews_forced = ews(None)
    df_ews_null = ews(model.parameter[model.parameter > bcrit].index[1])
            
    # Concatenate EWS DataFrames. Index [tsid, Variable, Time]
    return (df_ews_forced, df_ews_null) 

# Compute kendall tau values at equally spaced time points
def _ktau_series(series: pd.Series, t_res: Callable[[int], int], name: str) -> pd.Series:
    
    # Function to compute kendall tau for time seires data up to point t_fin
    def _ktau_compute(series: pd.Series):
        return stats.kendalltau(series.index.to_numpy(), series.to_numpy())[0]
    
    tVals = series.index[::t_res(len(series))]
    start = series[pd.notna(series)].index[1] # Get first non-NaN index in series
    # Return series
    return pd.Series((_ktau_compute(series.loc[start:t_end]) for t_end in tVals), index=tVals, name=name)

def ktau(df: pd.DataFrame, t_res: Callable[[int], int] = lambda ts_len: int(ts_len / 150)):

    def compute(tup: tuple[Any, pd.DataFrame]):
        (tsid, variable), df_tsid = tup
        df = pd.concat(
            (_ktau_series(df_tsid.set_index("Time")[col], t_res, name) for col, name in (("variance", "ktau_variance"), ("ac1", "ktau_ac"))),
            axis=1
        ).reset_index()
        df.insert(0, "tsid", tsid)
        df.insert(1, "Variable", variable)
        return df

    # Concatenate kendall tau dataframes
    return pd.concat(map(compute, df.groupby(["tsid", "Variable"])))

def may_fold(ts_len: int, sims: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    
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
    
def cr_trans(ts_len: int, sf: int = 4, sims: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        
    # Model parameters
    # sf = 4  # scale factor
    sigma_x = 0.01  # noise intensity
    sigma_y = 0.01
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
        sigma=[sigma_x, sigma_y],
        ts_len=ts_len,
        sims=sims,
    )
    
def cr_hopf(ts_len: int, sf: int = 4, sims: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:

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
        sigma=[sigma_x, sigma_y],
        ts_len=ts_len,
        sims=sims,
    )


import plotly.graph_objects as go

def plot_simulation(
    df_ews_forced: pd.DataFrame, 
    df_ews_null: pd.DataFrame, 
    df_ktau_forced: pd.DataFrame, 
    df_ktau_null: pd.DataFrame,
    df_ml_forced: pd.DataFrame, 
    df_ml_null: pd.DataFrame,
    bool_pred_early: bool = True
) -> tuple[go.Figure, pd.DataFrame, pd.DataFrame]:
    
    # --------
    # Import EWS and ML data
    # –------------

    # # Import EWS data
    # Use EWS data in x

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
        t_start = df["Time"].iloc[0]
        t_transition = (
            df[["Time", "residuals"]].dropna()["Time"].iloc[-1]
        )  # where the residuals end

        # Get prediction interval in time
        t_pred_start = t_start + (t_transition - t_start) * pred_interval_rel[0]
        t_pred_end = t_start + (t_transition - t_start) * pred_interval_rel[1]

        # Get ktau data specific to this variable and tsid.
        # Get data within prediction interval
        df_ktau_forced_final = df_ktau_forced[
            (df_ktau_forced["tsid"] == tsid)
            & (df_ktau_forced["Time"] >= t_pred_start)
            & (df_ktau_forced["Time"] <= t_pred_end)
        ]
        df_ml_forced_final = df_ml_forced[
            (df_ml_forced["tsid"] == tsid)
            & (df_ml_forced["Time"] >= t_pred_start)
            & (df_ml_forced["Time"] <= t_pred_end)
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
        t_start = df["Time"].iloc[0]
        t_transition = (
            df[["Time", "residuals"]].dropna()["Time"].iloc[-1]
        )  # where the residuals end

        # Get prediction interval in time
        t_pred_start = t_start + (t_transition - t_start) * pred_interval_rel[0]
        t_pred_end = t_start + (t_transition - t_start) * pred_interval_rel[1]

        # Get ktau data specific to this variable and tsid.
        # Get data within prediction interval
        df_ktau_null_final = df_ktau_null[
            (df_ktau_null["tsid"] == tsid)
            & (df_ktau_null["Time"] >= t_pred_start)
            & (df_ktau_null["Time"] <= t_pred_end)
        ]
        df_ml_null_final = df_ml_null[
            (df_ml_null["tsid"] == tsid)
            & (df_ml_null["Time"] >= t_pred_start)
            & (df_ml_null["Time"] <= t_pred_end)
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

    df_counts = pd.DataFrame({col.lower():counts[col] if col in counts.index else 0 for col in LSTM.COLUMNS})

    # --------------------
    # Functions to compute ROC
    # –--------------------


    # Function to compute ROC data from truth and indicator vals
    # and return a df.
    def roc_compute(truth_vals, indicator_vals):

        # Compute ROC curve and threhsolds using sklearn
        fpr, tpr, thresholds = metrics.roc_curve(truth_vals, indicator_vals)

        return pd.DataFrame({
            "fpr": fpr, 
            "tpr": tpr, 
            "thresholds": thresholds, 
            "auc": metrics.auc(fpr, tpr) # Compute AUC (area under curve)
        })


    # ---------------------
    ## Compute ROC data
    # –--------------------

    # Initialise list for ROC dataframes for predicting May fold bifurcation
    list_roc = []


    # Assign indicator and truth values for ML prediction
    indicator_vals = df_ml_preds["bif_prob"]
    truth_vals = df_ml_preds["truth value"]
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc["ews"] = "ML bif"
    list_roc.append(df_roc)


    # Assign indicator and truth values for variance
    indicator_vals = df_ktau_preds["ktau_variance"]
    truth_vals = df_ktau_preds["truth value"]
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc["ews"] = "Variance"
    list_roc.append(df_roc)


    # Assign indicator and truth values for lag-1 AC
    indicator_vals = df_ktau_preds["ktau_ac"]
    truth_vals = df_ktau_preds["truth value"]
    df_roc = roc_compute(truth_vals, indicator_vals)
    df_roc["ews"] = "Lag-1 AC"
    list_roc.append(df_roc)


    # Concatenate roc dataframes
    df_roc_full = pd.concat(list_roc, ignore_index=True)


    # -------------
    # Plotly fig
    # ----------------

    fig = go.Figure()
    df_roc = df_roc_full

    # ML bif plot
    df_trace = df_roc[df_roc["ews"] == "ML bif"]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            mode="lines",
            name="ML bif (AUC={})".format(df_trace.round(2)["auc"].iloc[0]),
        )
    )

    # Variance plot
    df_trace = df_roc[df_roc["ews"] == "Variance"]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            name="Variance (AUC={})".format(df_trace.round(2)["auc"].iloc[0]),
        )
    )

    # Lag-1  AC plot
    df_trace = df_roc[df_roc["ews"] == "Lag-1 AC"]
    fig.add_trace(
        go.Scatter(
            x=df_trace["fpr"],
            y=df_trace["tpr"],
            name="Lag-1 AC (AUC={})".format(df_trace.round(2)["auc"].iloc[0]),
        )
    )

    # Line y=x
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 100),
            y=np.linspace(0, 1, 100),
            showlegend=False,
            line={"color": "black", "dash": "dash"},
        )
    )

    fig.update_xaxes(
        title="False positive rate",
        range=[-0.01, 1],
    )
    fig.update_yaxes(
        title="True positive rate",
    )

    fig.update_layout(
        legend=dict(
            x=0.6,
            y=0,
        ),
        width=600,
        height=600,
        title="ROC for CR Hopf model",
    )
    
    return fig, df_counts, df_roc_full
    
def simulate_ml(lstm: LSTM, df: pd.DataFrame) -> pd.DataFrame:
    
    sims = {name:Dataset(
        name=f"ts{name[0]}-{name[1]}",
        df=uniq_df.set_index("Time").dropna(),
        feature_cols={"residuals":"Residuals"},
        age_format="",
    ) for name, uniq_df in df.groupby(["tsid", "Variable"])}
    
    for dataset in sims.values():
        lstm.run(dataset)
    
    list_df: list[pd.DataFrame] = []
    for (tsid, var), dataset in sims.items():
        df = lstm.results[dataset][1].reset_index()
        # Add info to dataframe
        df.insert(0, "tsid", tsid)
        df.insert(1, "Variable", var)
        
        list_df.append(df)
        
    return pd.concat(list_df).sort_values(
        ["Variable", "tsid", "Time"], na_position="first"
    )

type _Models = Iterable[tuple[str, int, tuple[pd.DataFrame, pd.DataFrame]]]
def test_models():
    for ts_len in (500, 1500):
        for test in (may_fold, cr_hopf, cr_trans):
            print("Simulating", test.__name__, "for length", ts_len)
            # Use only x (p0)
            df_ews_forced, df_ews_null = tuple(df[df["Variable"] == df["Variable"].iloc[0]] for df in test(ts_len=ts_len))
            yield test.__name__, ts_len, (df_ews_forced, df_ews_null)

# def from_datasets(datasets: Sequence[Dataset]) -> _Models:
    
#     def single(dataset: Dataset):
#         return dataset.df.rename({"Feature":"Variable", "Model":"tsid"}).reset_index()
        
#     return (single(dataset) for dataset in datasets)
    
def test(lstm: LSTM, output: str | None = None, models: _Models = test_models()):
    for name, ts_len, ews in models: 
        print("Computing ktau...")
        ktaus = [ktau(df) for df in ews]
        print("Running deep learning model on simulations...")
        mls = [simulate_ml(lstm, df) for df in ews]
        fig, _, _ = plot_simulation(
            ews[0],
            ews[1],
            ktaus[0],
            ktaus[1],
            mls[0],
            mls[1]
        )
        if output:
            fig.write_image(os.path.join(output, f"ROC {name} len{ts_len}.png"))
        
        
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(
                    prog='LSTM Training Data Generator',
                    description='Generates training data')
    parser.add_argument('modelpath', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('--extension', '-e', type=str)
    args = parser.parse_args()
    
    lstm = LSTMLoader(args.modelpath, extension=args.extension).with_args(verbose=False)
    
    test(lstm, output=args.output)
    

        