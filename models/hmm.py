from typing import Any, Literal, Optional, Self

import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM
from models import Model, Dataset

_Results = tuple[GaussianHMM, int, float, np.ndarray]
class HMM(Model[_Results]):
    
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

    def _best_model(
        self: Self,
        features, 
        n_regimes: int,
        n_tries=20
    ) -> tuple[GaussianHMM, Any, float, float, float]:
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
                    best_logL, best_model, best_features = float(logL), model, model_features
                    
            except ValueError:
                pass
        
        return (
            best_model, 
            best_features, 
            best_logL, 
            float(model.aic(best_features)), 
            float(model.bic(best_features))
        )
        
    # We need to find the model with the lowest BIC, so we test 10 (by default) models per number of regimes above 2, until the while loop stops (its criteria is commented below)
    def _best_model_of_unknown_regimes(
        self: Self,
        features,
    ) -> tuple[int, GaussianHMM, Any, float, float, float]:
        BIC_INDEX = 5
        results: list[tuple[int, GaussianHMM, Any, float, float, float]] = []
        
        # print("### Finding best number of regimes, k ###")
        
        # While we have less than two results or the BIC is not increasing we loop over models
        while len(results) < 2 or (results[-1][BIC_INDEX] is None or results[-2][BIC_INDEX] is None) or results[-1][BIC_INDEX] <= results[-2][BIC_INDEX]:
            n_regimes = len(results) + 2
            # Find the best model out of a number of models
            results.append(
                (n_regimes, ) + self._best_model(
                    features=features,
                    n_regimes=n_regimes,
                )
            )
            
        best = min(results, key=lambda tuple: tuple[BIC_INDEX])
                
        return best
    
    def run(
        self: Self,
        dataset: Dataset,
    ):
            
        ages = dataset.ages()
        features = dataset.features().to_numpy(copy=False)
        
        n_regimes = self.n_regimes
        if n_regimes is None:
            n_regimes, model, features, _, _, bic = self._best_model_of_unknown_regimes(
                features = features,
            )
        else:
            model, features, _, _, bic = self._best_model(
                features=features,
            )
            
        posterior = model.predict(features)
        # https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_casino.html for predicting n regimes
        
        self.results[dataset] = model, n_regimes, bic, posterior
            
    def _print(self: Self, dataset: Dataset):
        model, n_regimes, bic, posterior = self.results[dataset]
        ages = dataset.ages()
        
        c = "was unable to converge!" if not model.monitor_.converged else "was able to converge."
        print(f"The model has BIC {bic:.2f} and {c}") 
        
        print(f"Number of regimes: {n_regimes}")

        # print(f"Detected {len(accepted)} tipping points:")
        # for idx, state in accepted:
        #     print(f"Age {ages[idx]:.2f} with posterior value {posterior[idx, state]:.2f}")

    def _plot(self: Self, dataset: Dataset) -> Figure:
        model, n_regimes, bic, posterior = self.results[dataset]
        ages = dataset.ages()
        fig = plt.figure()
        axs = fig.subplots()
        axs.set_title(f"HMM {dataset.name} (Regimes={n_regimes})")
        axs.set_ylabel("Posterior probability")
        axs.set_xlabel(f"Age ({dataset.age_format})")
        for k in range(posterior.shape[1]):
            axs.plot(ages, posterior[:, k], label=f"P(state={k})")
            
        return fig