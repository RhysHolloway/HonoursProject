from typing import Any, Callable, Literal, Optional, Self, Union

import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from util import filter_points
from processing import Model, Dataset

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

class HMM(Model):
    
    def __init__(
        self: Self,
        n_regimes: Optional[int] = None,
        covariance_type: Literal["diag", "full"] ="full",
        n_iter=2000,
        min_duration_between_switches=20.0,
        min_state_duration=10.0,
        p_threshold: Optional[float] = 0.99,
        min_covar=1e-3,
        tol=1e-2,
    ):
        super().__init__("HMM")
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_duration_between_switches = min_duration_between_switches
        self.min_state_duration = min_state_duration
        self.p_threshold = p_threshold
        self.min_covar = min_covar
        self.tol = tol
        
    def fit_model(self: Self, features, n_regimes: int, random_state=None):
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            min_covar=self.min_covar,
            tol=self.tol,
            random_state=random_state,
        )
        model.fit(features)
        return (model, features)

    def best_model(
        self: Self,
        features, 
        n_regimes: int,
        n_tries=20
    ):
        best_model, best_features, best_logL = None, None, -np.inf
        for seed in range(n_tries):
            try:
                model, model_features = self.fit_model(
                    features=features, 
                    n_regimes=n_regimes,
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
        self: Self,
        features,
    ) -> list[tuple[int, Optional[float], Optional[float], Optional[GaussianHMM]]]:
        BIC_INDEX = 5
        results = []
        
        # print("### Finding best number of regimes, k ###")
        
        # While we have less than two results or the BIC is not increasing we loop over models
        while len(results) < 2 or (results[-1][BIC_INDEX] is None or results[-2][BIC_INDEX] is None) or results[-1][BIC_INDEX] <= results[-2][BIC_INDEX]:
            n_regimes = len(results) + 2
            # Find the best model out of a number of models
            results.append(
                (n_regimes, ) + self.best_model(
                    features=features,
                    n_regimes=n_regimes,
                )
            )
            
        best = min(results, key=lambda tuple: tuple[BIC_INDEX])
                
        return best
    
    def runner(
        self: Self,
        data: list[Dataset],
    ):
        
        def single(data: Dataset):
            
            ages = data.ages()
            features = data.features()
            
            n_regimes = self.n_regimes
            if n_regimes is None:
                n_regimes, model, features, _, _, bic = self.best_model_of_unknown_regimes(
                    features = features,
                )
            else:
                model, features, _, _, bic = self.best_model(
                    features=features,
                )
                
            states = model.predict(features)
            posterior = model.predict_proba(features)    
            
            switches = np.where(states[1:] != states[:-1])[0] + 1
            
            accepted = filter_points(
                points=switches,
                ages=ages,
                scores=posterior[switches, states[switches]],
                min_distance=self.min_duration_between_switches,
            )
            
            accepted = filter_accepted_ages(accepted, states, ages, posterior, self.p_threshold, self.min_state_duration)
            
            return data, model, n_regimes, bic, posterior, accepted
            
        results = [single(data) for data in data]
            
        def print_results(results):
            for result in results:
                dataset, model, n_regimes, bic, posterior, accepted = result
                ages = dataset.ages()
                
                c = "was unable to converge!" if not model.monitor_.converged else "was able to converge."
                print(f"The model has BIC {bic:.2f} and {c}") 
                
                print(f"Number of regimes: {n_regimes}")

                print(f"Detected {len(accepted)} tipping points:")
                for idx, state in accepted:
                    print(f"Age {ages[idx]:.2f} with posterior value {posterior[idx, state]:.2f}")

        def plot_single(result: tuple[Dataset, Any, int, float, np.ndarray]) -> Figure:
            dataset, model, n_regimes, bic, posterior, accepted = result
            ages = dataset.ages()
            fig = plt.figure()
            axs = fig.subplots()
            axs.set_title(f"HMM {dataset.name}")
            axs.set_ylabel("Posterior probability")
            axs.set_xlabel(f"Age ({dataset.age_format()})")
            for k in range(posterior.shape[1]):
                axs.plot(ages, posterior[:, k], label=f"P(state={k})")
            for a, state in accepted:
                axs.axvline(ages[a], linestyle="--", alpha=0.7)
                axs.text(ages[a], 1.1, f"{round(ages[a])} ({posterior[a, state]:.2f})", color='black', ha='center', va='bottom', rotation=90)
                
            return dataset, fig
        
        return (lambda: print_results(results), lambda: list(map(plot_single, results)))