#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:24:37 2026

Modified training data generation script for the tipping point-detecting deep learning model created by Thomas Bury

Made to run using only one file, multithreaded, and without AUTO-07p

@author: Rhys Holloway, Thomas Bury

"""
from sys import stderr
import abc
import builtins
import copy
import traceback
import atomics
from atomics.base import AtomicUint
import numpy as np
import ruptures
from typing import Any, Callable, Final, Literal, Self, Sequence, TypeAlias, Union
import pandas as pd
from joblib import Parallel, delayed
import scipy.optimize as opt
import scipy.optimize._nonlin as nl

from pycont.Types import ContinuationResult 
import pycont

from ..lstm import *
from .. import compute_residuals

class Counts(metaclass = abc.ABCMeta):
    
    @staticmethod
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
            setattr(self, Counts.attr(type), initializer(type))
    
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

def _init_atomic(_: BifType) -> AtomicUint:
    counter: AtomicUint = atomics.atomic(width=4, atype=atomics.UINT)
    counter.store(0)
    return counter

class _AtomicCounts(Counts):
    
    def __init__(self):
        super().__init__(_init_atomic)
    
    def _get(self, type: BifType) -> AtomicUint:
        return getattr(self, Counts.attr(type))
    
    def count(self, type: BifType) -> int:
        return self._get(type).load()
    
    def inc(self, type: BifType):
        self._get(type).inc()

class BifCounts(Counts):
    
    def __init__(self: Self, counts: Union[_AtomicCounts, None] = None):
        super().__init__(lambda type: 0 if counts is None else counts.count(type))
            
    def count(self: Self, type: BifType) -> int:
        return getattr(self, Counts.attr(type))
    
    def copy(self: Self) -> Self:
        return copy.copy(self)
        
    def __iadd__(self: Self, other: Counts):
        for type in bif_types():
            attr = Counts.attr(type)
            setattr(self, attr, getattr(self, attr) + other.count(type))

    def __add__(self: Self, other: Self) -> Self:
        new: Self = self.copy()
        new += other # type: ignore
        return new
    
    def inc(self, type: BifType):
        attr = Counts.attr(type)
        setattr(self, attr, getattr(self, attr) + 1)

_Simulation: TypeAlias = tuple[Sims, BifType]
_Simulations: TypeAlias = dict[int, _Simulation]

_ModelOutput: TypeAlias = list[_Simulation]

_2DArray: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]


def _needs_sample(counts: Counts, bif_type: BifType, bif_max: int) -> bool:
    return counts.count(bif_type) < bif_maximum(type=bif_type, bif_max=bif_max)


def _evaluate_polynomial(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    x, y = u
    x2 = x * x
    y2 = y * y
    xy = x * y

    f1 = (
        a[0] + a[1] * x + a[2] * y + a[3] * x2 + a[4] * xy +
        a[5] * y2 + a[6] * x * x2 + a[7] * x2 * y + a[8] * x * y2 +
        a[9] * y * y2
    )
    f2 = (
        b[0] + b[1] * x + b[2] * y + b[3] * x2 + b[4] * xy +
        b[5] * y2 + b[6] * x * x2 + b[7] * x2 * y + b[8] * x * y2 +
        b[9] * y * y2
    )
    return np.array([f1, f2])


def _jacobian(pars: _2DArray, u: np.ndarray) -> np.ndarray:
    x, y = u
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = pars[:10]
    b1,b2,b3,b4,b5,b6,b7,b8,b9,b10 = pars[10:]

    return np.array([
        [
            a2 + 2 * a4 * x + a5 * y + 3 * a7 * x * x + 2 * a8 * x * y + a9 * y * y,
            a3 + a5 * x + 2 * a6 * y + a8 * x * x + 2 * a9 * x * y + 3 * a10 * y * y,
        ],
        [
            b2 + 2 * b4 * x + b5 * y + 3 * b7 * x * x + 2 * b8 * x * y + b9 * y * y,
            b3 + b5 * x + 2 * b6 * y + b8 * x * x + 2 * b9 * x * y + 3 * b10 * y * y,
        ],
    ], dtype=float)


def _fixed_polynomial_field(pars: _2DArray) -> Callable[[np.ndarray], np.ndarray]:
    a = np.array(pars[:10], copy=True)
    b = np.array(pars[10:], copy=True)
    return lambda u: _evaluate_polynomial(a, b, u)


def _find_stable_equilibrium(pars: _2DArray, s0: _2DArray) -> tuple[np.ndarray, float] | None:
    field = _fixed_polynomial_field(pars)
    guesses = [
        np.array(s0, dtype=float, copy=True),
        np.zeros_like(s0, dtype=float),
        np.array(-s0, dtype=float, copy=True),
    ]

    for guess in guesses:
        try:
            sol = opt.root(field, guess, method="hybr", options={"xtol": 1e-10, "maxfev": 200})
        except (ValueError, OverflowError, FloatingPointError):
            continue

        if not sol.success or not np.all(np.isfinite(sol.x)):
            continue

        equilibrium = np.array(sol.x, dtype=float, copy=True)
        residual = field(equilibrium)
        if (not np.all(np.isfinite(residual))) or np.linalg.norm(residual) > 1e-8:
            continue

        eigvals = np.linalg.eigvals(_jacobian(pars, equilibrium))
        rightmost = float(np.max(np.real(eigvals)))
        if (not np.isfinite(rightmost)) or rightmost >= 0.0:
            continue

        return equilibrium, abs(rightmost)

    return None

def _gen_model(
    ts_len: int,
    bif_max: int,
    total_counts: _AtomicCounts,
    param_jobs: int,
    # Set a proportion (chosen randomly) of the parameters to zero.
    sparsity:float = 0.5
):

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
   
        
    # Generate parameters from normal distribution
    pars = np.random.normal(loc=0,scale=1,size=20)

    # Select a subset of parameters to be zero.
    pars[np.random.choice(range(len(pars)),int(len(pars)*sparsity),replace=False)] = 0
    
    # Negate high order terms to encourage boundedness of solns
    pars[6:10] = -abs(pars[6:10])
    pars[16:20] = -abs(pars[16:20])
    
    # Define intial condition (2 dimensional array)
    s0 = np.random.normal(loc=0,scale=2,size=2)

    equilibrium_data = _find_stable_equilibrium(pars, s0)
    if equilibrium_data is None:
        return

    equilibrium, rrate = equilibrium_data
    nonzero_parameters = np.flatnonzero(pars != 0.0)
    if len(nonzero_parameters) == 0:
        return

    def safe_simulate(par: int) -> _ModelOutput | Exception:
        try:
            return _simulate(
                par=int(par),
                pars=pars,
                equilibrium=equilibrium,
                rrate=rrate,
                ts_len=ts_len,
                bif_max=bif_max,
                total_counts=total_counts,
            )
        except Exception as exc:
            return exc

    results = Parallel(
        n_jobs=max(1, int(param_jobs)),
        backend="threading",
        return_as="generator_unordered",
        pre_dispatch=str(max(1, int(param_jobs))),
    )(delayed(safe_simulate)(int(par)) for par in nonzero_parameters)

    for r in results:
        if not total_counts < bif_max:
            break

        match r:
            case list(new_sims):
                yield new_sims
            case _:
                match type(r):
                    case builtins.ValueError:
                        if not ("Jacobian inversion yielded zero vector." in str(r) or "operands could not be broadcast together with shapes" in str(r)):
                            traceback.print_exception(r, file=stderr)
                        continue
                    case nl.NoConvergence | builtins.RecursionError | builtins.UnboundLocalError:
                        continue
                    case _:
                        traceback.print_exception(r, file=stderr)

SOLVER_PARAMETERS: Final[dict[str, Any]] = {
    "tolerance": 1e-8,
    "hopf_detection": True,
    "analyze_stability": False,
    "initial_directions": "increase_p",
    "limit_cycle_continuation": False,
    "recursive_branching": False,
    "param_min": -5.0,
    "param_max": 5.0,
}

def _make_polynomial_field(pars: _2DArray, par: int) -> Callable[[np.ndarray, float], np.ndarray]:
    coeffs = pars.copy()
    a = coeffs[:10]
    b = coeffs[10:]

    def set_parameter(value: float):
        if par < 10:
            a[par] = value
        else:
            b[par - 10] = value

    def field(u: np.ndarray, p: float) -> np.ndarray:
        set_parameter(p)

        x, y = u
        x2 = x * x
        y2 = y * y
        xy = x * y

        f1 = (
            a[0] + a[1] * x + a[2] * y + a[3] * x2 + a[4] * xy +
            a[5] * y2 + a[6] * x * x2 + a[7] * x2 * y + a[8] * x * y2 +
            a[9] * y * y2
        )
        f2 = (
            b[0] + b[1] * x + b[2] * y + b[3] * x2 + b[4] * xy +
            b[5] * y2 + b[6] * x * x2 + b[7] * x2 * y + b[8] * x * y2 +
            b[9] * y * y2
        )

        return np.array([f1, f2])

    return field


def _gen_branches_py(
        par: int,
        pars: _2DArray,
        equilibrium: _2DArray,
        steps: int = 500,
    ) -> tuple[Callable[[np.ndarray, float], np.ndarray], list[dict[str, Any]]]:
    field = _make_polynomial_field(pars, par)

    with np.errstate(invalid='ignore'):
        cont: ContinuationResult = pycont.arclengthContinuation(
            G=field, u0=equilibrium, p0=pars[par],
            ds_0=1e-02, ds_min=1e-03, ds_max=1e-01,
            n_steps=steps,
            solver_parameters=SOLVER_PARAMETERS,
            verbosity=pycont.Verbosity.OFF,
        )

    if len(cont.branches) == 0:
        return field, []

    branch = cont.branches[0]
    event = branch.termination_event
    if event is None or event.kind not in BIFS:
        return field, []

    return field, [{
        "kind": event.kind,
        "crit": event.p,
    }]

def _simulate(
        par: int,
        pars: _2DArray,
        equilibrium: _2DArray,
        rrate: float,
        ts_len: int,
        bif_max: int,
        total_counts: _AtomicCounts
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

    # Noise amplitude
    sigma_tilde = 0.01

    # Only simulate bifurcation types that have count below bif_max
    sims: _ModelOutput = []
    
    # Define length of simulation
    # This is 200 points longer than ts_len
    # to increase the chance that we can extract ts_len data points prior to a transition
    # (transition can occur before the bifurcation is reached)
    series_len = ts_len + 200
    
    null_run = False
    de_fun, branches = _gen_branches_py(par, pars, equilibrium)

    for vals in branches:
        if not total_counts < bif_max:
            break

        needs_forced = _needs_sample(total_counts, (vals["kind"], False), bif_max)
        needs_null = not null_run and _needs_sample(total_counts, (vals["kind"], True), bif_max)

        if not (needs_forced or needs_null):
            continue
             
        #-------------------
        ## Simulate models
        #------------------
            
        # Construct noise as in Methods
        rv_tri = np.random.triangular(0.75,1,1.25)
        sigma = np.sqrt(2 * rrate) * sigma_tilde * rv_tri
        
        for null in ((False, True) if not null_run else (False,)):
            
            bif_type: BifType = (vals["kind"], null)
            
            if _needs_sample(total_counts, bif_type, bif_max):
                b_start=pars[par]
                b_end=vals["crit"]
                
                null_location=0.0 if null else None
                
                # If a null simulation, simulate at a fixed value of b, given by b_null
                if null_location is not None:
                    assert 0.0 <= null_location and null_location <= 1.0
                    # Set binit and bcrit for simulation as equal to b_null
                    b_start += null_location*(b_end-b_start)
                    b_end = b_start
                
                # Pick sample spacing randomly from [0.1,0.2,...,1]
                dt_sample = np.random.choice(np.arange(1,11)/10)
                
                simulator = StochSim(
                    b_start=b_start, 
                    b_end=b_end, 
                    t_end=series_len,
                    dt_sample=dt_sample,
                )
                
                sim = simulator.simulate(
                    de_fun=de_fun,
                    s0=equilibrium,
                    sigma=sigma,
                )['p0']
                
                valid_positions = np.flatnonzero(sim.notna().to_numpy())
                if len(valid_positions) == 0:
                    continue
                last_valid_pos = int(valid_positions[-1])
                if last_valid_pos + 1 < ts_len:
                    continue

                valid_sim = sim.iloc[:last_valid_pos + 1]
                
                # Detect a jump to another state (in time-series prior to infinity jump)
                # Break points - higher penalty means less likely to detect jumps, get first jump
                t_jump: int = ruptures.Window(width=10, model="l2", jump=1, min_size=2).fit(
                    compute_residuals(valid_sim, span=0.2).to_numpy().reshape(-1, 1)
                ).predict(pen=1, n_bkps=1)[0] - 1 # type: ignore
                
                # Output minimum of tnan or tjump
                transit_pos = min(last_valid_pos, t_jump)
                
                if transit_pos + 1 >= ts_len:
                    start = transit_pos + 1 - ts_len
                    sims.append((
                        # Simulations
                        pd.Series(sim.iloc[start:transit_pos + 1], index=pd.Index(np.arange(ts_len), name="time")),
                        # Label
                        bif_type,
                    ))
                    
                    if not null_run and null:
                        null_run = True
                        break
    
                
    return sims

def _create_groups(
    df_labels: pd.DataFrame,
    bif_max: int,
    validation_percentage: float = 0.04,
    test_percentage: float = 0.01,    
) -> Groups:
    for type in bif_types():
        max = bif_maximum(type=type, bif_max=bif_max)
        matches = df_labels[["bif", "null"]].eq(type).all(axis=1)
        df_labels = df_labels.drop(df_labels.index[matches & (matches.astype(int).cumsum() > max)])

    #----------------------------
    # Create groups file in ratio for training:validation:testing
    #-----------------------------

    # Create the file groups.csv with headers (sequence_ID, dataset_ID)
    # Use numbers 1 for training, 2 for validation and 3 for testing
    # Use ratio 38:1:1

    # Compute number of bifurcations for each group
    num_validation = int(np.floor(bif_max*validation_percentage))
    num_test = int(np.floor(bif_max*test_percentage))
    num_train = bif_max - num_validation - num_test

    # Create list of group numbers
    group_nums = [1]*num_train + [2]*num_validation + [3]*num_test
    
    # Set group numbers for each classification
    def to_group(classifier: pd.DataFrame) -> pd.Series:
        assert bif_max == len(classifier)
        return pd.Series(group_nums, index=classifier.index, name = "dataset_ID")
    
    return pd.concat((to_group(classifier) for _, classifier in df_labels.groupby('class_label')), sort=True).to_frame()

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
    param_jobs: int | None = None,
    ) -> TrainData:
    
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
                pd.DataFrame(columns=LABEL_COLS).to_csv(labels_path, index_label=INDEX_COL)
            
            labels = pd.read_csv(labels_path, index_col=INDEX_COL)
            
            for seq_id, bif, null in labels[['bif', 'null']].itertuples(index=True, name=None):
                biftype: BifType = (bif, null)
                if counts.count(biftype) < bif_maximum(biftype, bif_max):
                    
                    sim = pd.read_csv(os.path.join(path, f"sims/tseries{seq_id}.csv"), index_col="time")["p0"]
                    
                    if len(sim) != ts_len:
                        raise RuntimeError(f"Batch {batch_num} loaded simulation {seq_id} with non-matching length {len(sim)}! (expected {ts_len})")
                    
                    simulations[seq_id] = (sim, biftype)
                    counts.inc(biftype)
                          
            label_file = open(labels_path, "a+")
                
        except Exception as e:
            traceback.print_exception(e)
            
    print("Batch", batch_num, "loaded", len(simulations), "previous simulations")
    
    current = max(simulations.keys()) + 1 if len(simulations) != 0 else 1
    if param_jobs is None:
        param_jobs = os.cpu_count() or 1

    while counts < bif_max:
        added: set[BifType] = set()
        
        for results in _gen_model(
            ts_len=ts_len,
            bif_max=bif_max,
            total_counts=counts,
            param_jobs=param_jobs,
        ):
            for result in results:
                biftype = result[1]
                already_added = any((t, True) in added for t in BIFS) if biftype[1] else biftype in added
                if _needs_sample(counts, biftype, bif_max) and not already_added:
                    simulations[current] = result
                    counts.inc(biftype)
                    added.add(biftype)
                    if label_file is not None and path is not None:
                        result[0].to_csv(os.path.join(path, f"sims/tseries{current}.csv"))
                        label_file.write(f"{current},{_num_from_label(biftype)},{biftype[0]},{biftype[1]}\n")
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
        ), columns = [INDEX_COL] + LABEL_COLS,
    ).set_index(INDEX_COL)
    
    groups = _create_groups(df_labels, bif_max)
    
    if path is not None:
        df_labels.to_csv(os.path.join(path, "labels.csv"))
        groups.to_csv(os.path.join(path, "groups.csv"))
    
    print("Batch", batch_num, "finished")    
    return sims, df_labels, groups

# Returns (sims, labels, groups)
def multibatch(
        batches: Sequence[int], 
        ts_len: int,
        bif_max: int,
        path: Union[str, None] = None,
    ) -> TrainData:
        
    if any(b <= 0 for b in batches):
        raise ValueError("Input batch numbers contains a value less than or equal to 0!")
    
    if len(batches) == 0:
        raise ValueError("Please provide a non empty batch list!")

    if len(batches) > 1:
        batch_jobs = min(len(batches), os.cpu_count() or 1)
        param_jobs = max(1, (os.cpu_count() or 1) // batch_jobs)
        return combine_batches(Parallel(n_jobs=batch_jobs, backend="loky")(
            delayed(batch)(
                batch_num=batch_num,
                ts_len=ts_len,
                bif_max=bif_max,
                path=path,
                param_jobs=param_jobs,
            )
            for batch_num in batches
        )) # type: ignore
    else:
        return batch(
            batch_num=batches[0],
            ts_len=ts_len, 
            bif_max=bif_max,
            path=path,
            param_jobs=os.cpu_count() or 1,
        )
    
_DEFAULT_BIF_MAX = 1000
def run_with_args(
    ts_len: int,
    batches: Sequence[int],
    bif_max: int = _DEFAULT_BIF_MAX,
    output: str | None = None
):
    return multibatch(
        batches=batches, 
        ts_len=ts_len, 
        bif_max=bif_max,
        path=output,
    )
    
def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Training Data Generator',
                    description='Generates training data')
    parser.add_argument('output', type=str)
    parser.add_argument('-b', '--batches', type=int, default=1)
    parser.add_argument('-s', '--batch-start', type=int, default=1)
    parser.add_argument('-l', '--length', type=int)
    parser.add_argument('-m', '--bifurcations', type=int, default=_DEFAULT_BIF_MAX)
    args = parser.parse_args()

    run = run_with_args
    if __name__ == "__main__" and __spec__ is not None and __spec__.name != __name__:
        import importlib
        run = importlib.import_module(__spec__.name).run_with_args

    run(ts_len=args.length, batches=list(range(args.batch_start, args.batch_start + args.batches)), bif_max=args.bifurcations, output=args.output)

if __name__ == "__main__":
    _main()
