#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:24:37 2026

Modified training data generation script for the tipping point-detecting deep learning model created by Thomas Bury

Made to run using only one file, multithreaded, and without AUTO-07p

@author: Rhys Holloway, Thomas Bury

"""

import abc
import builtins
import copy
import traceback
import atomics
import numpy as np
import ruptures
from typing import Any, Callable, Final, Self, Sequence, Union
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import concurrent.futures
from concurrent.futures import Executor, Future, ThreadPoolExecutor
import scipy.optimize._nonlin as nl
import scipy.integrate
import pycont
import warnings

from ..lstm import *

def _generate_simulation(
    # Define convergence threshold
    conv_thresh = 1e-8, 
    # Set a proportion (chosen randomly) of the parameters to zero.
    sparsity=0.5
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.float64]:
    
    """
    Created on Mon Jul 15 15:24:37 2019

    @author: Thomas Bury

    Script to generate dynamical systems with random parameters and a stable equi:
        Generate random parameters
        Simulate dyanmical system
        Check convergence
        If yes, output parameters and equilibrium
        
        Dynamical system
        dx/dt = a1 + a2 x + a3 y + a4 x^2 + a5 xy + a6 y^2 + a7 x^3 + a8 x^2 y + a9 x y^2 + a10 y^3
        dy/dt = b1 + b2 x + b3 y + b4 x^2 + b5 xy + b6 y^2 + b7 x^3 + b8 x^2 y + b9 x y^2 + b10 y^3
        

    """
    
    # Stop when system with convergence found
    while True:
        
        # Generate parameters from normal distribution
        pars = np.random.normal(loc=0,scale=1,size=20)

        # Select a subset of parameters to be zero.
        pars[np.random.choice(range(len(pars)),int(len(pars)*sparsity),replace=False)] = 0
        
        # Negate high order terms to encourage boundedness of solns
        pars[6:10] = -abs(pars[6:10])
        pars[16:20] = -abs(pars[16:20])
        
        # Define intial condition (2 dimensional array)
        s0 = np.random.normal(loc=0,scale=2,size=2)
        
        # Define timesteps to evaluate at
        t = np.arange(0., 100, 0.01)
        
        # Define derivative function
        def f(_t: np.float64, s: np.ndarray[np.float64]):
            '''
            s: 2D state vector
            a: 10D vector of parameters for x dynamics
            b: 10D vecotr of parameterrs for y dyanmics
            '''
            x = s[0]
            y = s[1]
            # Polynomial forms up to third order
            polys = np.array([1,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3])
            # Output
            dydt = np.array([np.dot(pars[:10],polys), np.dot(pars[10:],polys)])
            
            return dydt
        
        points = scipy.integrate.odeint(f, s0, t,
                                full_output=False,
                                hmin=1e-14,
                                mxhnil=0, printmessg=False, tfirst=True)
        
        # Put into pandas
        df_traj = pd.DataFrame(points, index=t, columns=['x','y'])
        
        # Does the sysetm blow up?
        # Does the system contain Nan?
        # Does the system contain inf?
        if df_traj.abs().max().max() > 1e3 or df_traj.isna().values.any() or np.isinf(df_traj.values).any():
            continue
        
        # Does the system converge?
        # Difference between max and min of last 10 data points
        diff = df_traj.iloc[-10:-1].max() - df_traj.iloc[-10:-1].min()
        # L2 norm
        norm = np.sqrt(np.square(diff).sum())
        if norm > conv_thresh:
            continue
        
        break
    
    # If made it this far, system is good for bifurcation continuation
    # print('System converges - export equilibria and parameter values\n')
    # Export equilibrim data
    equi = points[-1]

    ## Compute the dominant eigenvalue of this system at equilbrium
    
    # Compute the Jacobian at the equilibrium
    x, y =equi
    [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]=pars[:10]
    [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]=pars[10:]
    
    # df/dx
    j11 = a2 + 2*a4*x + a5*y + 3*a7*x**2 + 2*a8*x*y + a9*y**2
    # df/dy
    j12 = a3 + 2*a6*y + a5*x + 3*a10*y**2 + 2*a9*x*y + a8*x**2
    # dg/dx
    j21 = b2 + 2*b4*x + b5*y + 3*b7*x**2 + 2*b8*x*y + b9*y**2
    # dg/dy
    j22 = b3 + 2*b6*y + b5*x + 3*b10*y**2 + 2*b9*x*y + b8*x**2

    # Assign component to Jacobian
    jac = np.array([[j11,j12],[j21,j22]])
    
    # Compute eigenvalues
    evals = np.linalg.eigvals(jac)
    
    # Compute the real part of the dominant eigenvalue (smallest magnitude)
    rrate: np.float64 = abs(max(lam.real for lam in evals))

    return (pars, equi, rrate)

class _Counts(metaclass = abc.ABCMeta):
    
    def attr(type: BifType) -> str:
        match type:
            case ("HB", True):
                return "null_h_count"
            case ("LP", True):
                return "null_f_count"
            case ("BP", True):
                return "null_b_count"
            case ("HB", False):
                return "hopf_count"
            case ("LP", False):
                return "fold_count"
            case ("BP", False):
                return "branch_count"
    
    def __init__(self: Self, initializer: Callable[[BifType], Any]):
        for type in bif_types():
            setattr(self, _Counts.attr(type), initializer(type))
    
    @abc.abstractmethod
    def count(self: Self, type: BifType) -> int:
        pass
    
    @abc.abstractmethod
    def inc(self: Self, type: BifType):
        pass
    
    def null_count(self: Self) -> int:
        return sum(self.count((bif, True)) for bif in BIFS)
        
    def bif_count(self: Self) -> int:
        return sum(self.count((bif, False)) for bif in BIFS)
    
    def total(self: Self) -> int:
        return self.bif_count() + self.null_count()
    
    def __lt__(self: Self, bif_max: int) -> bool:
        return any(self.count(type) < bif_maximum(type, bif_max) for type in bif_types())
    
    def __str__(self: Self) -> str:
        return ", ".join((("Null " if null else "") + type + ": " + str(self.count((type, null)))) for type, null in bif_types())

class _AtomicCounts(_Counts):
    
    def __init__(self):
        super().__init__(lambda _: atomics.atomic(width=4, atype=atomics.INT))
    
    def _get(self, type: BifType) -> atomics.INT:
        return getattr(self, _Counts.attr(type))
    
    def count(self, type: BifType) -> int:
        return self._get(type).load()
    
    def inc(self, type: BifType):
        self._get(type).inc()

class BifCounts(_Counts):
    
    def __init__(self: Self, counts: Union[_AtomicCounts, None] = None):
        super().__init__(lambda type: 0 if counts is None else counts.count(type))
            
    def count(self: Self, type: BifType) -> int:
        return getattr(self, _Counts.attr(type))
    
    def copy(self: Self) -> Self:
        return copy.copy(self)
        
    def __iadd__(self: Self, other: _Counts):
        for type in bif_types():
            attr = _Counts.attr(type)
            setattr(self, attr, getattr(self, attr) + other.count(type))

    def __add__(self: Self, other: Self) -> Self:
        new = self.copy()
        new += other
        return new
    
    def inc(self, type: BifType):
        attr = _Counts.attr(type)
        setattr(self, attr, getattr(self, attr) + 1)

SOLVER_PARAMETERS: Final[dict] = {
    "tolerance": 1e-8,
    "hopf_detection": True,
    "param_min": -5.0,
    "param_max": 5.0,
}

type _Simulation = tuple[Sims, BifType]
type _Simulations = dict[int, _Simulation]

type _ModelOutput = list[_Simulation]
def _gen_model(
        ts_len: int,
        bif_max: int,
        total_counts: _AtomicCounts,
        pool: Executor,
        simulations: _ModelOutput = [],
) -> _ModelOutput:
    
    model_counts = _AtomicCounts()
    
    # Multithread the parameter simuations
    tasks = [
        pool.submit(
            _simulate, 
            par, 
            pars, 
            equi, 
            rrate, 
            ts_len, 
            bif_max, 
            total_counts, 
            model_counts
        ) for par in range(len(pars)) if pars[par] != 0.0
    ]
    
    with warnings.catch_warnings(action="ignore", category=scipy.integrate.ODEintWarning):
        pars, equi, rrate = _generate_simulation()

    def retry():
        for task in tasks:
            task.cancel()
        return _gen_model(ts_len, bif_max, total_counts=total_counts, pool=pool, simulations=simulations) 

    for task in concurrent.futures.as_completed(tasks):
        e = task.exception()
        r = e if e is not None else task.result()
        match r:
            case list(new_sims):
                simulations += new_sims
            case _:
                match type(r):
                    case builtins.ValueError:
                        if "Jacobian inversion yielded zero vector." in str(r):
                            return retry()
                    case nl.NoConvergence | builtins.RecursionError:
                        return retry()
                    case concurrent.futures.CancelledError | builtins.UnboundLocalError:
                        pass
                    case _:
                        traceback.print_exception(r)
    
    for task in tasks:
        task.cancel()

    return simulations

def _gen_branches_py(
        par: int,
        pars: np.ndarray[np.float64],
        equi: np.ndarray[np.float64],
        steps: int = 500,
    ) -> list[dict]:
    pars = pars.copy()
    initial = pars[par]
    
    def G(u: np.ndarray[np.float64], p: np.float64):
        pars[par] = p
        
        x, y = u
        
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = pars[:10]
        b1,b2,b3,b4,b5,b6,b7,b8,b9,b10 = pars[10:]
        
        f1 = a1 + a2*x + a3*y + a4*x**2 + a5*x*y + a6*y**2 + a7*x**3 + a8*x**2*y + a9*x*y**2 + a10*y**3
        f2 = b1 + b2*x + b3*y + b4*x**2 + b5*x*y + b6*y**2 + b7*x**3 + b8*x**2*y + b9*x*y**2 + b10*y**3
        
        return np.array([f1, f2])
       
    with np.errstate(invalid='ignore'):
        cont = pycont.arclengthContinuation(
            G=G, u0=equi, p0=initial,
            ds_0=1e-02, ds_min=1e-03, ds_max=1e-01,
            n_steps=steps,
            solver_parameters=SOLVER_PARAMETERS,
            verbosity=pycont.Verbosity.OFF,
        )

    return [{
        'type': branch.termination_event.kind,
        'value': branch.termination_event.p, 
        'branch_vals':branch.termination_event.u,
        'initial_param':initial,
    } for branch in cont.branches if branch.termination_event is not None and branch.termination_event.kind in BIFS]      

def _simulate(
        par: int,
        pars: np.ndarray[np.float64],
        equi: np.ndarray[np.float64],
        rrate: np.float64,
        ts_len: int,
        bif_max: int,
        total_counts: _AtomicCounts,
        model_counts: _AtomicCounts,
    ) -> _ModelOutput:
    
    """
    Created on Wed Jul 31 10:05:17 2019

    @author: tbury

    SCRIPT TO:
    Get info from bifurcation output
    Run stochastic simulations up to bifurcation points for ts_len+200 time units
    Detect transition point using change-point algorithm
    Ouptut time-series of ts_len time units prior to transition

    """   
     
    branches = _gen_branches_py(par, pars, equi)
    pars = pars.copy()

    # Noise amplitude
    sigma_tilde = 0.01
        
    #-------------------
    ## Simulate models
    #------------------
        
    # Construct noise as in Methods
    rv_tri = np.random.triangular(0.75,1,1.25)
    # rv_tri = 1 # temporary
    sigma = np.sqrt(2*rrate) * sigma_tilde * rv_tri

    # Only simulate bifurcation types that have count below bif_max
    sims: _ModelOutput = []
    
    # Define length of simulation
    # This is 200 points longer than ts_len
    # to increase the chance that we can extract ts_len data points prior to a transition
    # (transition can occur before the bifurcation is reached)
    series_len = ts_len + 200
    
    def de_fun(s: np.ndarray[np.float64], b: float) -> np.ndarray[np.float64]:
            pars[par] = b
            '''
            Input:
            s is state vector
            pars is dictionary of parameter values
            
            Output:
            array [dxdt, dydt]
            
            '''
            
            # Polynomial forms up to third order
            x: np.float64=s[0]
            y: np.float64=s[1]
            polys: np.ndarray[np.float64] = np.array([1.0,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3])

            dxdt: np.float64 = np.dot(pars[10:], polys)
            dydt: np.float64 = np.dot(pars[:10], polys)
                        
            return np.array([dxdt, dydt])
    
    # Pick sample spacing randomly from [0.1,0.2,...,1]
    dt_sample = np.random.choice(np.arange(1,11)/10)
        
    for null in [False, True]:
        
        binit=model["initial_value"]
        bcrit=model['value']
        
        null_location=0.0 if null else None
        
        # If a null simulation, simulate at a fixed value of b, given by b_null
        if null_location is not None:
            assert 0.0 <= null_location and null_location <= 1.0
            # Set binit and bcrit for simulation as equal to b_null
            binit += null_location*(bcrit-binit)
            bcrit = binit
        
        simulator = StochSim(
            binit=binit, 
            bcrit=bcrit, 
            ts_len=series_len,
            dt_sample=dt_sample,
        )
        
        for model in branches:
            
            bif_type: BifType = (model['type'], null)
            
            if (
                total_counts.count(bif_type) < bif_maximum(type=bif_type, bif_max=bif_max) and
                (model_counts.null_count() if null else model_counts.count(bif_type)) == 0 # Has this model not ran this (or another null sim if null) simulation yet
            ):
                df_out = simulator.simulate(
                    de_fun=de_fun,
                    s0=equi, 
                    sigma=sigma, 
                )
                
                if df_out is None:
                    continue
                
                trans_time = _trans_detect(df_out)
                        
                if trans_time > ts_len and (not null or model_counts.null_count() == 0) and model_counts._get(bif_type).cmpxchg_strong(0, 1).success:
                    df_cut = df_out.loc[trans_time-ts_len:trans_time-1].reset_index()
                    # Have time-series start at time t=0
                    df_cut["time"] = df_cut["time"]-df_cut["time"][0]
                    df_cut.set_index("time", inplace=True)
                    sims.append((
                        # Simulations
                        df_cut['x'],
                        # Label
                        bif_type,
                    ))
                    break
    
    return sims
    
# Function to detect change points
def _trans_detect(df_in: pd.DataFrame):
    '''
    Function to detect a change point in a time series
    Input:
        df_in: DataFrame indexed by time, with series data in column 'x'
    Output:
        float: time at which transition occurs
    '''
    # Check for a jump to Nan
    df_nan: pd.DataFrame = df_in[np.isnan(df_in['x'])]
    if df_nan.size == 0:
        # Assign a big value to t_nan
        t_nan = 1e6
    else:
        # First time of Nan
        t_nan = df_nan.iloc[0].name
    
    
    if t_nan > 500:        
        # Detect a jump to another state (in time-series prior to infinity jump)
        
        # First detrend the series (breakpoint detection is not working well with non-stationary data)
        span = 0.2
        series_data: pd.Series = df_in.loc[:t_nan-1]['x']
        smooth_data = lowess(series_data.values, series_data.index.values, frac=span)[:,1]
        # On rare occasion the smoothing function messes up
        # In this case output 0, and run new simulation
        if len(series_data.values) != len(smooth_data):
            return 0
        
        # Compute residuals
        residuals = series_data.values[:len(smooth_data)] - smooth_data
        resid_series = pd.Series(residuals, index=series_data.index)
    
        
        array_traj = resid_series.values.transpose()
        # Window-based change point detection
        algo = ruptures.Window(width=10, model="l2", jump=1, min_size=2).fit(array_traj)
        # Break points - higher penalty means less likely to detect jumps
        bps = algo.predict(pen=1)
        
        t_jump = bps[0]
    else:
        # Assign big value to t_jump
        t_jump = 1e6
    
    # Output minimum of tnan or tjump
    out = min(max(0,t_nan-1),t_jump-1)
    return out

def _create_groups(
    df_labels: pd.DataFrame,
    bif_max: int,
    validation_percentage = 0.04,
    test_percentage = 0.01,    
) -> Groups:
    for type in bif_types():
        max = bif_maximum(type=type, bif_max=bif_max)
        matches = df_labels[["bif", "null"]].eq(type).all(axis=1)
        df_labels = df_labels.drop(df_labels.index[matches & (matches.cumsum() > max)])

    #----------------------------
    # Create groups file in ratio for training:validation:testing
    #-----------------------------

    # Create the file groups.csv with headers (sequence_ID, dataset_ID)
    # Use numbers 1 for training, 2 for validation and 3 for testing
    # Use raito 38:1:1

    # Compute number of bifurcations for each group
    num_validation = int(np.floor(bif_max*validation_percentage))
    num_test = int(np.floor(bif_max*test_percentage))
    num_train = bif_max - num_validation - num_test

    # Create list of group numbers
    group_nums = [1]*num_train + [2]*num_validation + [3]*num_test
    
    # Set group numbers for each classification
    def to_group(classifier: pd.DataFrame) -> pd.DataFrame:
        assert bif_max == len(classifier)
        classifier = classifier[[INDEX_COL]]
        classifier['dataset_ID'] = group_nums
        return classifier
    
    return pd.concat((to_group(classifier) for _, classifier in df_labels.groupby('class_label'))).sort_values(by=[INDEX_COL])

################################################################################

import os.path
import os

def _num_from_label(t: BifType) -> int:
    if t[1]:
        return 3
    else:
        match t[0]:
            case "LP":
                return 0
            case "HB":
                return 1
            case "BP":
                return 2    

def batch(
    batch_num: int, 
    ts_len: int, 
    bif_max: int,
    path: Union[None, str] = None,
    ) -> TrainData:
    
    def generator():
        pool = ThreadPoolExecutor()
        if pool._max_workers // 10 > 0:
            tasks: list[Future[_ModelOutput]] = []
            while counts < bif_max:
                while len(tasks) < pool._max_workers // 10:
                    tasks.append(
                        pool.submit(
                            _gen_model,
                            ts_len=ts_len,
                            bif_max=bif_max,
                            total_counts=counts,
                            pool=pool,
                        )
                    )
                concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)
                for task in tasks:
                    if task.done():
                        match task.exception():
                            case None:
                                tasks.remove(task)
                                yield task.result()
                            case e:
                                match type(e):
                                    case concurrent.futures.CancelledError:
                                        pass
                                    case _:
                                        traceback.print_exception(e)
            for task in tasks:
                task.cancel()
        else:
            while counts < bif_max:
                yield _gen_model(
                        ts_len=ts_len,
                        bif_max=bif_max,
                        total_counts=counts,
                        pool=pool,
                    )
    
    print("Batch", batch_num, "start")
    
    simulations: _Simulations = dict()
    counts = _AtomicCounts()
    
    # Continue from previous state
    label_file = None
    if path is not None:
        path = os.path.join(path, f"batch{batch_num}/")
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, f"sims/"), exist_ok=True)
        try:
            labels_path = os.path.join(path, "labels.csv")
            
            if not os.path.exists(labels_path):
                pd.DataFrame(columns=LABEL_COLS).to_csv(labels_path, index=False)
            
            labels = pd.read_csv(labels_path)
            
            for seq_id, bif, null in labels[[INDEX_COL, 'bif', 'null']].itertuples(index=False, name=None):
                type: BifType = (bif, null)
                if counts.count(type) < bif_maximum(type, bif_max):
                    simulations[seq_id] = (
                        pd.read_csv(os.path.join(path, f"sims/tseries{seq_id}.csv")),
                        (bif, null)
                    )
                    counts.inc(type)
                
            if len(simulations) > 0:
                # ts_len is same between previous iterations and now
                assert len(next(iter(simulations.values()))[0]) == ts_len
                          
            label_file = open(labels_path, "a+")
                
        except Exception as e:
            traceback.print_exception(e)
            
    print("Batch", batch_num, "loaded", len(simulations), "previous simulations")
    
    current = max(simulations.keys()) + 1 if len(simulations) != 0 else 1
    
    for results in generator():
        added = set()
        for result in results:
            type = result[1]
            if counts.count(type) < bif_maximum(type=type, bif_max=bif_max) and not (any((t, True) in added for t in BIFS) if type[1] else type in added):
                simulations[current] = result
                counts.inc(type)
                added.add(type)
                if label_file is not None:
                    result[0].to_csv(os.path.join(path, f"sims/tseries{current}.csv"))
                    label_file.write(f"{current},{_num_from_label(type)},{type[0]},{type[1]}\n")
                current += 1

        if len(added) != 0:
            print("Batch", batch_num, "-", counts)
            if label_file is not None:
                label_file.flush()
    
    print("Batch", batch_num, "finished generating")
        
    sims = {i:sims for i, (sims, _) in simulations.items()}

    df_labels: pd.DataFrame = pd.DataFrame(
        (
            (i, _num_from_label(label), label[0], label[1])
            for i, (_, label) in simulations.items()
        ), columns = LABEL_COLS,
    )
    
    df_groups = _create_groups(df_labels, bif_max)
    
    if path is not None:
        df_labels.to_csv(os.path.join(path, "labels.csv"), index=False)
        df_groups.to_csv(os.path.join(path, "groups.csv"), index=False)
    
    print("Batch", batch_num, "finished")    
    return sims, df_labels, df_groups, counts

# Returns (sims, labels, groups)
def multibatch(
        batches: Sequence[int], 
        ts_len: int,
        bif_max: int,
        batch_pool: Executor, 
        path: Union[str, None] = None,
    ) -> TrainData:
        
    if any(b <= 0 for b in batches):
        raise ValueError("Input batch numbers contains a value less than or equal to 0!")
    
    if len(batches) == 0:
        raise ValueError("Please provide a non empty batch list!")

    if len(batches) > 1:
        return combine_batches(batch_pool.map(
            batch, 
            batches, 
            [ts_len] * len(batches), 
            [bif_max] * len(batches), 
            [path] * len(batches),
        ))  
    else: 
        return batch(
            batch_num=batches[0],
            ts_len=ts_len, 
            bif_max=bif_max,
            path=path,
        )
    
def run_with_args():
    
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Training Data Generator',
                    description='Generates training data')
    parser.add_argument('output', type=str)
    parser.add_argument('-b', '--batches', type=int, default=1)
    parser.add_argument('-s', '--batch-start', type=int, default=1)
    parser.add_argument('-l', '--length', type=int)
    parser.add_argument('-m', '--bifurcations', type=int, default=1000)
    args = parser.parse_args()
    
    batches = list(range(args.batch_start, args.batch_start + args.batches))
    
    import sys
    import threading
            
    class SelectOutput():
        out = sys.stdout

        def write(self, text):
            if threading.current_thread().name == "printer":
                self.out.write(text)

        def flush(self):
            self.out.flush()
            
    sys.stdout = SelectOutput()
        
    def set_printer_thread():
        threading.current_thread().name = "printer"
        
    pool = concurrent.futures.ProcessPoolExecutor(initializer=set_printer_thread)
    
    if len(batches) == 1:
        set_printer_thread()

    import atexit
    atexit.register(lambda: pool.shutdown(wait=False, cancel_futures=True))

    multibatch(
        batch_pool=pool, 
        batches=batches, 
        ts_len=args.length, 
        bif_max=args.bifurcations,
        path=args.output,
    )
    
if __name__ == "__main__":
    run_with_args()
