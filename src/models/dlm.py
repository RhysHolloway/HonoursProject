import numpy as np
import matplotlib.pyplot as plt
from pybats.analysis import *
from pybats.plot import *

from processing import Dataset, Model

class DLM(Model):    
    
    def __init__(self):
        super().__init__("DLM")    

    def runner(
        self,
        data: list[Dataset]
    ):
        
        # forecast_end = max(data.ages())
        
        # mod, samples = analysis(features, ages, family="normal",
        #     # forecast_start=forecast_start,      # First time step to forecast on
        #     forecast_end=forecast_end,          # Final time step to forecast on
        #     # k=1,                                # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
        #     # prior_length=6,                     # How many data point to use in defining prior
        #     # rho=.5,                             # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
        #     # deltrend=0.95,                      # Discount factor on the trend component (the intercept)
        #     # delregn=0.95                        # Discount factor on the regression component
        # )
        
        # model = sm.tsa.MarkovRegression(features[:,1], k_regimes=3)
        # results = model.fit()
            
        # states = model.predict(features)
        # posterior = model.predict_proba(features)    
        
        # switches = np.where(states[1:] != states[:-1])[0] + 1
        
        # accepted = filter_points(
        #     points=switches,
        #     ages=ages,
        #     scores=posterior[switches, states[switches]],
        #     min_distance=min_duration_between_switches,
        # )
        
        # accepted = filter_accepted_ages(accepted, states, ages, posterior, p_threshold, min_state_duration)
            
        def print_results():  
        
            # print(mod)
            print()
            
            # c = "was unable to converge!" if not model.monitor_.converged else "was able to converge."
            # print(f"The model has BIC {bic:.2f} and {c}") 
            
            # print(f"Number of regimes: {n_regimes}")

            # print(f"Detected {len(accepted)} tipping points:")
            # for idx, state in accepted:
            #     print(f"Age {ages[idx]:.2f}{age_format} with posterior value {posterior[idx, state]:.2f}")

        def plot_results(fig):
            
            # forecast = np.median(samples)                                  
            # forecast_start, forecast_end, k = 0,0,20
            # # Plot the 1-step ahead point forecast plus the 95% credible interval
            # ax = fig.subplots(1,1)   
            # ax.plot(samples[:,0,0])
            # # ax = plot_data_forecast(fig, ax, features[forecast_start:forecast_end + k], forecast, samples,
            #                         dates=np.arange(forecast_start, forecast_end+1, dtype='int'))
            # ax = ax_style(ax, ylabel='Sales', xlabel='Time', xlim=[forecast_start, forecast_end],
            #             legend=['Forecast', 'Sales', 'Credible Interval'])
            
            # pass
            # ax = fig.subplots(1,1)
            # ax = plot_data_forecast(fig, ax,
            #             data_1step.Sales,
            #             median(samples_1step),
            #             samples_1step,
            #             data_1step.index,
            #             credible_interval=75)
            # plt.title(f"Tipping points for {name} using {model_name}", pad=80)
            # plt.ylabel("Posterior probability")
            # for k in range(posterior.shape[1]):
            #     plt.plot(ages, posterior[:, k], label=f"P(state={k})")
            # for a, state in accepted:
            #     plt.axvline(ages[a], linestyle="--", alpha=0.7)
            #     plt.text(ages[a], 1.1, f"{round(ages[a])}{age_format} ({posterior[a, state]:.2f})", color='black', ha='center', va='bottom', rotation=90)
            pass
        
        return (print_results, plot_results)