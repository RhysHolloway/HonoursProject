#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import numpy as np
import scipy
import ruptures
from functools import reduce
from typing import Final
import ewstools
import pandas as pd
import pycont
import scipy.integrate as spi
from statsmodels.nonparametric.smoothers_lowess import lowess
import atomics
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
# from util import get_project_path

def conv() -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.float64]:
    # Stop when system with convergence found
    while True:
        
        # Generate parameters from normal distribution
        pars = np.random.normal(loc=0,scale=1,size=20)
        
        # Set a proportion (chosen randomly) of the parameters to zero.
        # Draw sparsity from uniform rv
        sparsity=0.5
        index_zero = np.random.choice(range(20),int(20*sparsity),replace=False)
        pars[index_zero]=0
        #    # Try setting a1=b1=0
        #    pars[0]=0
        #    pars[10]=0
        # Negate high order terms to encourage boundedness of solns
        pars[6:10] = -abs(pars[6:10])
        pars[16:20] = -abs(pars[16:20])
        
        pars_a = pars[:10]
        pars_b = pars[10:]    
        
        # Define intial condition (2 dimensional array)
        s0 = np.random.normal(loc=0,scale=2,size=2)
        
        # Define derivative function
        def f(s,t0,a,b):
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
            dydt = np.array([np.dot(a,polys), np.dot(b,polys)])
            
            return dydt
        
        
        ## Simulate the system
        t = np.arange(0., 100, 0.01)
        s, info_dict = spi.odeint(f, s0, t, args=(pars_a,pars_b),
                                full_output=True,
                                hmin=1e-14,
                                mxhnil=0, printmessg=False)
        
        # Put into pandas
        df_traj = pd.DataFrame(s, index=t, columns=['x','y'])
        
        # Does the sysetm blow up?
        if df_traj.abs().max().max() > 1e3:
            print('System blew up - run new model')
            continue
        
        # Does the system contain Nan?
        if df_traj.isna().values.any():
            print('System contains Nan value - run new model')
            continue
            
        # Does the system contain inf?
        
        if np.isinf(df_traj.values).any():
            print('System contains Inf value - run new model')
            continue
        
        # Does the system converge?
        # Difference between max and min of last 10 data points
        diff = df_traj.iloc[-10:-1].max() - df_traj.iloc[-10:-1].min()
        # L2 norm
        norm = np.sqrt(np.square(diff).sum())
        # Define convergence threshold
        conv_thresh = 1e-8
        if norm > conv_thresh:
            print('System does not converge - run new model')
            continue
        
        break
    
    # If made it this far, system is good for bifurcation continuation
    # print('System converges - export equilibria and parameter values\n')
    # Export equilibrim data
    equi = np.array(df_traj.iloc[-1].values)
    pars = np.concatenate([pars_a,pars_b])


    ## Compute the dominant eigenvalue of this system at equilbrium
    
    # Compute the Jacobian at the equilibrium
    x=equi[0]
    y=equi[1]
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

    return (equi, pars, rrate)

class Counts:
    hopf_count=atomics.atomic(width=4, atype=atomics.INT)
    fold_count=atomics.atomic(width=4, atype=atomics.INT)
    branch_count=atomics.atomic(width=4, atype=atomics.INT)
    null_h_count=atomics.atomic(width=4, atype=atomics.INT)
    null_f_count=atomics.atomic(width=4, atype=atomics.INT)
    null_b_count=atomics.atomic(width=4, atype=atomics.INT)
    

    def null_count(self) -> int:
        return self.null_b_count.load() + self.null_h_count.load() + self.null_f_count.load()
        
    def bif_count(self) -> int:
        return self.hopf_count.load() + self.fold_count.load() + self.branch_count.load()
    
    def total(self) -> int:
        return self.bif_count() + self.null_count()
    
    def less_than(self, bif_max: int) -> bool:
        return self.hopf_count.load() < bif_max or self.fold_count.load() < bif_max or self.branch_count.load() < bif_max or () < bif_max
    
    # def add(self, other):
    #     self.hopf_count.g.add(other.hopf_count.fetch())
    #     self.fold_count+=other.fold_count
    #     self.branch_count+=other.branch_count
    #     self.null_count+=other.null_count
    #     self.null_h_count+=other.null_h_count
    #     self.null_f_count+=other.null_f_count
    #     self.null_b_count+=other.null_b_count
        
    
    # def add_tuple(self, other: list[tuple]):
    #     for t in other:
    #         self.add(t[0])
    #     return [t[1:] for t in other]
        

# ode = pycobi.ODESystem.from_file(get_project_path("src/training/model.f90"))

# PAR_IDS = [r for r in range(1, 22) if r != 11]
SOLVER_PARAMETERS: Final[dict] = {
    "tolerance": 1e-6,
    "hopf_detection": True,
    "param_min": -5.0,
    "param_max": 5.0,
}
ALLOWED_BIFS: Final[set[pycont.Types.EventKind]] = set(["BP", "LP", "HB"])

def gen_bifs(
        par: int, 
        pars: np.ndarray,
        equi: np.ndarray,
) -> list[dict]:
    
    # eta_sols, _ = ode.run(
    #         origin="init", starting_point='EP1', name=f'par{par}', bidirectional=True,
    #         NDIM=2, IPS=1, IRS=0, ILP=1,
    #         ICP=par, 
    #         NTST=15, NCOL=4, IAD=3, ISP=1, ISW=1, IPLT=0, NBC=0, NINT=0,
    #         NMX=500, NPR=500,  MXBF=  -1, IID =   2, ITMX= 8, ITNW= 5, NWTN= 3, JAC= 0,
    #         EPSL= 1e-07, EPSU = 1e-07, EPSS = 1e-05,
    #         DS  =   1e-02, DSMIN=  1e-03, DSMAX= 1e-01, IADS=   1,
    #         NPAR = 20, THL =  {}, THU =  {},
    #         UZSTOP =  {1:[-5,5]}
    #     )
     
    pars = pars.copy()
    
    def G(u, p):
        pars[par] = p
        
        x, y = u[0], u[1]
        
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = pars[:10]
        b1,b2,b3,b4,b5,b6,b7,b8,b9,b10 = pars[10:]
        
        f1 = a1 + a2*x + a3*y + a4*x**2 + a5*x*y + a6*y**2 + a7*x**3 + a8*x**2*y + a9*x*y**2 + a10*y**3
        f2 = b1 + b2*x + b3*y + b4*x**2 + b5*x*y + b6*y**2 + b7*x**3 + b8*x**2*y + b9*x*y**2 + b10*y**3
        
        return np.array([f1, f2])
       
    initial = pars[par]
       
    cont = pycont.arclengthContinuation(
        G, equi, initial,
        ds_0=1e-02, ds_min=  1e-03, ds_max= 1e-01,
        n_steps=500,
        solver_parameters=SOLVER_PARAMETERS,
        verbosity=pycont.Verbosity.OFF,
    )

    return [{
        'type': branch.termination_event.kind,
        'value': branch.termination_event.p, 
        'branch_vals':branch.termination_event.u,
        'initial_param':initial,
        'param':par,
    } for branch in cont.branches if branch.termination_event is not None and branch.termination_event.kind in ALLOWED_BIFS]

# TODO: restart batch sim if this function overflows
def sim_model(model: dict, pars: np.ndarray[np.float64], equi: np.ndarray[np.float64], dt_sample=1, series_len=500, sigma=0.1, null_sim=False, null_location=0):
    '''
    Function to run a stochastic simulation of model up to bifurcation point
    Input:
        model (class) : contains details of model to simulate
        dt_sample : time between sampled points (must be a multiple of 0.01)
        series_len : number of points in time series
        sigma (float) : amplitude factor of GWN - total amplitude also
            depends on parameter values
        null_sim (bool) : Null simulation (bifurcation parameter fixed) or
            transient simulation (bifurcation parameter increments to bifurcation point)
        null_location (float) : Value in [0,1] to determine location along bifurcation branch
            where null is simulated. Value is proportion of distance to the 
            bifurcation from initial point (0 is initial point, 1 is bifurcation point)
    Output:
        DataFrame of trajectories indexed by time
    '''

    
    # Simulation parameters
    dt = 0.01
    t0 = 0
    tburn = 100 # burn-in period
#   seed = 0 # random number generation 
    
    # Bifurcation point of model
    bcrit = model['value'] # bifurcation point
    
    # Initial value of bifurcation parameter
    bl = model['initial_param']
    
    # If a null simulation, simulate at a fixed value of b, given by b_null
    if null_sim:
        b_null = bl + null_location*(bcrit-bl)
        # Set bl and bh for simulation as equal to b_null
        bl = b_null
        bh = b_null
    
    # If a transient simulation, let b go from bl to bh, where bh is bifurcation point
    else:
        bh = bcrit
    
    s0 = equi    # initial condition

    # Model equations

    def de_fun(s:np.ndarray[np.float64]) -> np.ndarray[np.float64]:
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
        polys = np.array([1.0,x,y,x**2,x*y,y**2,x**3,x**2*y,x*y**2,y**3])
        
        dxdt: np.float64 = np.dot(pars[10:], polys)
        dydt: np.float64 = np.dot(pars[:10], polys)
                      
        return np.array([dxdt, dydt])
        
        
    
    # Initialise arrays to store single time-series data
    t = np.arange(t0, series_len*dt_sample, dt)
    s: np.ndarray[np.ndarray[np.float64]] = np.zeros([len(t),2])
    
   
    # Set up bifurcation parameter b, that increases linearly in time from bl to bh
    b = pd.Series(np.linspace(bl,bh,len(t)),index=t)   

    ## Implement Euler Maryuyama for stocahstic simulation
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = [int(tburn/dt),2])
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = [len(t/dt),2])
    
    # Run burn-in period on s0
    for i in range(int(tburn/dt)):
        s0 = s0 + de_fun(s0)*dt + dW_burn[i]
        
    # Initial condition post burn-in period
    s[0] = s0
    
    # Run simulation
    for i in range(len(t)-1):
        # Update bifurcation parameter
        pars[model['param']] = b.iloc[i]
        s[i+1] = s[i] + de_fun(s[i])*dt + dW[i]
            
    # Store series data in a DataFrame
    data = {'Time': t,'x': s[:,0],'y': s[:,1], 'b':b.values}
    df_traj = pd.DataFrame(data)
    
    # Filter dataframe according to spacing
    df_traj_filt = df_traj.iloc[0::int(dt_sample/dt)].copy()
    
    # Replace time column with integers for compatibility
    # with trans_detect
    df_traj_filt['Time'] = np.arange(0,series_len)
    df_traj_filt.set_index('Time', inplace=True)

    return df_traj_filt
    
# Function to detect change points
def trans_detect(df_in):
    '''
    Function to detect a change point in a time series
    Input:
        df_in: DataFrame indexed by time, with series data in column 'x'
    Output:
        float: time at which transition occurs
    '''
    # Check for a jump to Nan
    df_nan = df_in[np.isnan(df_in['x'])]
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
        series_data = df_in.loc[:t_nan-1]['x']
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
        model = "l2"
        algo = ruptures.Window(width=10, model=model, jump=1, min_size=2).fit(array_traj)
        # Break points - higher penalty means less likely to detect jumps
        bps = algo.predict(pen=1)
        
        t_jump = bps[0]
    else:
        # Assign big value to t_jump
        t_jump = 1e6
    
    # Output minimum of tnan or tjump
    out = min(max(0,t_nan-1),t_jump-1)
    return out
    
    
def stoch_sims(
    counts: Counts,
    pars: np.ndarray[np.float64],
    equi: np.ndarray[np.float64],
    rrate: np.float64,
    bifs: list[dict],
    bif_max,
    ts_len,
) -> list[tuple[pd.DataFrame, int]]:
    """
    Created on Wed Jul 31 10:05:17 2019

    @author: tbury

    SCRIPT TO:
    Get info from b.out files (AUTO files)
    Run stochastic simulations up to bifurcation points for ts_len+200 time units
    Detect transition point using change-point algorithm
    Ouptut time-series of ts_len time units prior to transition

    """

    hopf_sim = True
    fold_sim = True
    branch_sim = True
    null_h_sim = True
    null_f_sim = True
    null_b_sim = True

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
    sims: list[tuple[pd.DataFrame, int]] = []
    
    for model in bifs:
        
        def sim(type: str, count: atomics.INT, max: int, null_sim: bool, label: int):
            if model['type'] == type and count.load() < max:
                df_out = sim_model(model, pars, equi, dt_sample=dt_sample, series_len=series_len,
                            sigma=sigma, null_sim=null_sim)
                trans_time = trans_detect(df_out)
                # Only if trans_time > ts_len, keep and cut trajectory
                if trans_time > ts_len:
                    if count.fetch_inc() < max:
                        df_cut = df_out.loc[trans_time-ts_len:trans_time-1].reset_index()
                        # Have time-series start at time t=0
                        df_cut['Time'] = df_cut['Time']-df_cut['Time'][0]
                        df_cut.set_index('Time', inplace=True)
                        sims.append((df_cut[['x']], label))
                        return True
                    else:
                        count.store(max)
            return False
        
        # Pick sample spacing randomly from [0.1,0.2,...,1]
        dt_sample = np.random.choice(np.arange(1,11)/10)
        # Define length of simulation
        # This is 200 points longer than ts_len
        # to increase the chance that we can extract ts_len data points prior to a transition
        # (transition can occur before the bifurcation is reached)
        series_len = ts_len + 200
        
        if null_h_sim and sim('HB', counts.null_h_count, bif_max // 3, True, 3):
            null_h_sim = False
        
        # Simulate a null_f trajectory
        if null_f_sim and sim('LP', counts.null_f_count, bif_max//3, True, 3):
            null_f_sim = False

            
        # Simulate a null_b trajectory
        if null_b_sim and sim('BP', counts.null_b_count, bif_max-2*(bif_max//3), True, 3):
            null_b_sim = False

        # Simulate a Hopf trajectory
        if hopf_sim and sim('HB', counts.hopf_count, bif_max, False, 1):
            hopf_sim = False
                
        # Simulate a Fold trajectory
        if fold_sim and sim('LP', counts.fold_count, bif_max, False, 0):
            fold_sim = False

        if branch_sim and sim('BP', counts.branch_count, bif_max, False, 2):
            branch_sim = False

    return sims

def to_traindata(
    counts: Counts,
    simulations: list[list[tuple[pd.DataFrame, pd.DataFrame, int]]]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    
    bif_total = counts.total()
    
    sims: list[pd.DataFrame] = [
        s
        for run in simulations
        for s,_,_ in run
    ]
    
    resids: list[pd.DataFrame] = [
        r 
        for run in simulations
        for _,r,_ in run
    ]
        
    #----------------------------
    # Convert label files into single csv file
    #-----------------------------

    df_labels: pd.DataFrame = pd.DataFrame(
        {
            'sequence_ID': [i] * len(run), 
            'class_label': [label for _,_,label in run]
        } for i, run in enumerate(simulations))
    

    #----------------------------
    # Create groups file in ratio for training:validation:testing
    #-----------------------------

    # Create the file groups.csv with headers (sequence_ID, dataset_ID)
    # Use numbers 1 for training, 2 for validation and 3 for testing
    # Use raito 38:1:1        

    # Collect Fold bifurcations (label 0)
    df_fold = df_labels[df_labels['class_label']==0].copy()
    # Collect Hopf bifurcations (label 1)
    df_hopf = df_labels[df_labels['class_label']==1].copy()
    # Collect Branch points (label 2)
    df_branch = df_labels[df_labels['class_label']==2].copy()
    # Collect Null labels (label 3)
    df_null = df_labels[df_labels['class_label']==3].copy()

    # Check they all have the same length
    assert len(df_fold) == len(df_hopf)
    assert len(df_hopf) == len(df_branch)
    assert len(df_branch) == len(df_null)


    # Compute number of bifurcations for each group
    num_valid = int(np.floor(bif_total*0.04))
    num_test = int(np.floor(bif_total*0.01))
    num_train = bif_total - num_valid - num_test

    # Create list of group numbers
    group_nums = [1]*num_train + [2]*num_valid + [3]*num_test

    # Assign group numbers to each bifurcation category
    df_fold['dataset_ID'] = group_nums
    df_hopf['dataset_ID'] = group_nums
    df_branch['dataset_ID'] = group_nums
    df_null['dataset_ID'] = group_nums

    # Concatenate dataframes and select relevant columns
    df_groups = pd.concat([df_fold,df_hopf,df_branch,df_null])[['sequence_ID','dataset_ID']]
    # Sort rows by sequence_ID
    df_groups.sort_values(by=['sequence_ID'], inplace=True)
    
    return sims, resids, df_labels, df_groups

def compute_resids(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    return [
        ewstools.core.ews_compute(
                df['x'],
                smooth = 'Lowess',
                span = 0.2,
                ews=[]
            )['EWS metrics'][['Residuals']] 
        for df in dfs]

################################################################################

from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

tp = ThreadPoolExecutor()
def gen_data(counts: Counts, equi, pars, rrate, bif_max, batch_num, ts_len) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    print("START BIF DATA {}", batch_num)
    bifs: list[list[dict]] = tp.map(gen_bifs, range(len(pars)), [pars] * len(pars), [equi] * len(pars))    
    bifs = [val for sublist in bifs for val in sublist]
    
    if len(bifs) == 0:
        print("SIM COULD NOT CONVERGE", batch_num)

    print("STOCH SIMS BIF DATA {}", batch_num)
    sims, labels = zip(*stoch_sims(
        counts,
        pars,
        equi,
        rrate,
        bifs,
        bif_max,
        ts_len,
    ))
        
    logger.debug("RESIDS BIF DATA {}", batch_num)
    resids: list[pd.DataFrame] = compute_resids(sims)
    print("END BIF DATA {}", batch_num)    
    return zip(sims, resids, labels)

pool = ProcessPoolExecutor()

def kill():
    tp.shutdown(wait=False, cancel_futures=True)
    pool.shutdown(wait=False, cancel_futures=True)
import atexit
atexit.register(kill)

def batch(batch_num: int, ts_len: int, bif_max: int) -> tuple:
    print("Start batch {}", batch_num)    

    tasks: list[Future[list[tuple[pd.DataFrame, pd.DataFrame, int]]]] = []
    
    equi, pars, rrate = conv()
    counts = Counts()
    
    simulations: list[list[tuple[pd.DataFrame, pd.DataFrame, int]]] = []
    
    while counts.less_than(bif_max):
        while len(tasks) < 3:
            tasks.append(pool.submit(gen_data, counts, equi, pars, rrate, bif_max, batch_num, ts_len))
        concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)
        for task in tasks:
            if task.done():
                if task.exception() is None:
                    tasks.remove(task)
                    simulations.append(task.result())
                else:
                    print(task.exception())
                    for task in tasks:
                        task.cancel()
                    match task.exception():
                        case scipy.optimize.NoConvergence as e:
                            batch(batch_num, ts_len, bif_max)
                        case _:
                            pass
                    raise task.exception()
    for task in tasks:
        simulations.append(task.result())
        
    tuple = to_traindata(counts, batch_num, simulations)
    
    print("Finished batch {}", batch_num)    
    return tuple

def multibatch(batches: int, ts_len: int, bif_max: int):
    
    def concat(a: tuple, b: tuple) -> tuple:
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
    
    return reduce(concat, pool.map(batch, range(1, batches + 1), [ts_len] * batches, [bif_max] * batches))

if __name__ == "__main__":
    np.seterr(all='ignore')
    print(multibatch(2, 50, 10))