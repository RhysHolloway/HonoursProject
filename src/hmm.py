from typing import Any, Callable, Literal, Optional, Union

import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from util import filter_points

def fit_model(features, n_regimes, covariance_type, n_iter, min_covar, tol, random_state = None):       
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        min_covar=min_covar,
        tol=tol,
        random_state=random_state,
    )
    model.fit(features)
    return (model, features)

def best_model(
    features, 
    n_regimes,
    covariance_type: Union[Literal["diag"], Literal["full"]], 
    n_iter, 
    min_covar, 
    tol, 
    n_tries=20
):
    best_model, best_features, best_logL = None, None, -np.inf
    for seed in range(n_tries):
        try:
            model, model_features = fit_model(
                features=features, 
                n_regimes=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                min_covar=min_covar,
                tol=tol,
                random_state=seed,
            )
            
            logL = model.score(model_features)
            
            if logL > best_logL:
                best_logL, best_model, best_features = logL, model, model_features
                
        except ValueError:
            pass
    
    return (
        best_model, 
        best_features, 
        best_logL, 
        model.aic(best_features), 
        model.bic(best_features)
    )

# We need to find the model with the lowest BIC, so we test 10 (by default) models per number of regimes above 2, until the while loop stops (its criteria is commented below)
def best_model_of_unknown_regimes(
    features, 
    covariance_type: Union[Literal["diag"], Literal["full"]], 
    n_iter, 
    min_covar, 
    tol
) -> list[tuple[int, Optional[float], Optional[float], Optional[GaussianHMM]]]:
    BIC_INDEX = 5
    results = []
    
    # print("### Finding best number of regimes, k ###")
    
    # While we have less than two results or the BIC is not increasing we loop over models
    while len(results) < 2 or (results[-1][BIC_INDEX] is None or results[-2][BIC_INDEX] is None) or results[-1][BIC_INDEX] <= results[-2][BIC_INDEX]:
        n_regimes = len(results) + 2
        # Find the best model out of a number of models
        results.append(
            (n_regimes, ) + best_model(
                features=features,
                n_regimes=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                min_covar=min_covar,
                tol=tol,
            )
        )
           
    best = min(results, key=lambda tuple: tuple[BIC_INDEX])
    
    # for k, _, _, _, aic, bic in results: 
    #     best_aic = " (Lowest AIC)" if min(results, key=lambda tuple: tuple[BIC_INDEX])[0] == k else ""
    #     best_bic = " (Lowest BIC)" if best[0] == k else ""
    #     print(f"{k} regimes has AIC {aic} and BIC {bic}{best_aic}{best_bic}")
    
    # c = "was unable to converge!" if not best[1].monitor_.converged else "converged."
    # print(f"### The best model has {best[0]} regimes, and {c} ###")    
            
    return best

def filter_accepted_ages(prev_accepted, states, ages, post, p_threshold: Optional[float], min_state_duration):
    accepted = []

    for idx in prev_accepted:
        age = ages[idx]
        new_state = states[idx]

        # enforce minimum duration in the new state (in years)
        # find how long we stay in new_state starting at idx
        j = idx
        while j < len(states) and states[j] == new_state:
            j += 1

        end_age = ages[j-1] if j > idx else ages[idx]

        if end_age - age >= min_state_duration and (p_threshold is None or post[idx, new_state] >= p_threshold):
            accepted.append((idx, new_state))

    return accepted

def hmm(
    n_regimes: Optional[int] = None,
    covariance_type: str ="full",
    n_iter=2000,
    min_duration_between_switches=20.0,
    min_state_duration=10.0,
    p_threshold: Optional[float] = 0.99,
    min_covar=1e-3,
    tol=1e-2,
) -> Callable[[Any, Any, bool], None]:
    
    def run(
        ages, 
        features,
        n_regimes: Optional[int] = n_regimes
    ):
        if n_regimes is None:
            n_regimes, model, features, _, _, bic = best_model_of_unknown_regimes(
                features = features,
                covariance_type=covariance_type,
                n_iter=n_iter,
                min_covar=min_covar,
                tol=tol,
            )
        else:
            model, features, _, _, bic = best_model(
                features=features,
                n_regimes=n_regimes,
                covariance_type=covariance_type,
                n_iter=n_iter,
                min_covar=min_covar,
                tol=tol
            )
            
        states = model.predict(features)
        posterior = model.predict_proba(features)    
        
        switches = np.where(states[1:] != states[:-1])[0] + 1
        
        accepted = filter_points(
            points=switches,
            ages=ages,
            scores=posterior[switches, states[switches]],
            min_distance=min_duration_between_switches,
        )
        
        accepted = filter_accepted_ages(accepted, states, ages, posterior, p_threshold, min_state_duration)
            
        def print_results(age_format: str):  
            
            c = "was unable to converge!" if not model.monitor_.converged else "was able to converge."
            print(f"The model has BIC {bic:.2f} and {c}") 
            
            print(f"Number of regimes: {n_regimes}")

            print(f"Detected {len(accepted)} tipping points:")
            for idx, state in accepted:
                print(f"Age {ages[idx]:.2f}{age_format} with posterior value {posterior[idx, state]:.2f}")

        def plot_results(age_format: str):
            plt.ylabel("Posterior probability")
            for k in range(posterior.shape[1]):
                plt.plot(ages, posterior[:, k], label=f"P(state={k})")
            for a, state in accepted:
                plt.axvline(ages[a], linestyle="--", alpha=0.7)
                plt.text(ages[a], 1.1, f"{round(ages[a])}{age_format} ({posterior[a, state]:.2f})", color='black', ha='center', va='bottom', rotation=90)
        
        return (print_results, plot_results)
    
    return ("HMM", run)