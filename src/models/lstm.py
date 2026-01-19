import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import find_peaks
from util import filter_points, get_project_path, resample_df
from wrapper import Dataset, Model
from tensorflow.keras.models import load_model
import ewstools
# import ewstools

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Dropout, MaxPooling1D
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam, Optimizer

# def train_lstm_from_data(
#     data,
#     seq_len: int,
#     train: Union[int, float],
#     epochs: int,
#     patience: int = 30,
#     batch_size: int = 32,
#     kernel_initializer: str = 'lecun_normal',
#     optimizer: Optimizer = Adam(learning_rate = 0.0005),
# ):

#     if float(train) <= 0.0:
#         raise ValueError("Negative or zero training value provided!")
    
#     n_train: int = int(min(train, 1.0) * len(ages)) if isinstance(train, float) else max(int(train), len(ages))

#     Y_train, X_train = ages[:n_train], seq_features[:n_train].reshape(n_train, features.shape[1], 1)
    
#     Y_val, X_val = ages[n_train:], seq_features[n_train:].reshape(-1, features.shape[1], 1)
    
#     model = Sequential([
#         Input(shape=(features.shape[1], 1)),
#         Conv1D(
#             filters=50, 
#             kernel_size=12,
#             padding="same",
#             activation="relu",
#             kernel_initializer=kernel_initializer,
#         ),
#         Dropout(0.1),
#         MaxPooling1D(pool_size = 2),
#         LSTM(50, return_sequences=True, kernel_initializer = kernel_initializer),
#         Dropout(0.1),
#         LSTM(10, kernel_initializer = kernel_initializer),
#         Dense(4, activation='softmax', kernel_initializer = kernel_initializer)
#     ])

#     model.compile(
#         loss=SparseCategoricalCrossentropy(), 
#         optimizer=optimizer, 
#         metrics=['accuracy']
#     )
    
#     history = model.fit(
#         x=X_train,
#         y=Y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
#         validation_data=(X_val, Y_val),
#         verbose=True,
#     )

# Make sure to do linear interpolation before using
class LSTM(Model):
    
    def __init__(self):
        super().__init__("CNN-LSTM")
        
        root_path = get_project_path("env/deep-early-warnings-pnas/dl_train/best_models_tf215/")
        self.__500models = [load_model(root_path + "len500/" + name) for name in os.listdir(root_path + "len500/") if name[-6:] == ".keras"]
        self.__1500models = [load_model(root_path + "len1500/" + name) for name in os.listdir(root_path + "len1500/") if name[-6:] == ".keras"]
        
    def __compute_ews(dataset: Dataset, dic_bandwith: dict) -> list[pd.DataFrame]:

        # # Bandwidth sizes for Gaussian kernel (used in Dakos (2008) Table S3)
        # dic_bandwidth = {
        #     "End of greenhouse Earth": 25,
        #     "End of Younger Dryas": 100,
        #     "End of glaciation I": 25,
        #     "Bolling-Allerod transition": 25,
        #     "End of glaciation II": 25,
        #     "End of glaciation III": 10,
        #     "End of glaciation IV": 50,
        #     "Desertification of N. Africa": 10,
        # }
            
        #     return (model, history)
        # EWS computation parameters
        rw = 0.5  # half the length of the data

        # Loop through each record

        list_df = []
        for feature in dataset.feature_cols:
            
            df = dataset.df

            # Make time negative so it increaes up to transition
            df[dataset.age_col] = -df[dataset.age_col]
            # Series for computing EWS
            series = df.set_index(dataset.age_col)[feature]

            ts = ewstools.TimeSeries(series)
            ts.detrend(method="Gaussian")#, bandwidth=)
            ts.compute_var(rolling_window=rw)
            ts.compute_auto(rolling_window=rw, lag=1)

            df_ews = ts.state.join(ts.ews)
            list_df.append(df_ews)

        # Concatenate dataframes
        df_ews = pd.concat(list_df)

        # # Export
        return df_ews
        
    def __predict(self, dataset: Dataset):
        
        df_ews_forced = self.__compute_ews(dataset)
        
        series = df_ews_forced["residuals"]
        
        ts = ewstools.TimeSeries(dataset.features())
        # space predictions apart by 10 data points (inc must be defined in terms of time)
        dt = series.index[1] - series.index[0]
        inc = dt * 10
        
        classifier_list = self.__1500models if len(series) > 500 else self.__500models
        
        for i, classifier in enumerate(classifier_list):
            ts.apply_classifier_inc(classifier, inc=inc, verbose=1, name=str(i))

        # Get ensemble dl prediction
        dl_preds_mean = ts.dl_preds.groupby("time")[[0, 1, 2, 3]].mean()
        dl_preds_mean.columns = ["fold_prob", "hopf_prob", "branch_prob", "null_prob"]
        dl_preds_mean.index.name = "Time"
        
        return dl_preds_mean


    def runner(
        self,
        datasets: list[Dataset],
        # threshold: Union[float, list[float]] = [0.95, 0.99],
        prominence: float = 0.05,
        distance: int = 10,
    ):
        # thresholds: float = threshold if isinstance(threshold, list) else [threshold]
    
        predictions = [self.__predict(dataset) for dataset in datasets]

            # height = np.percentile(predictions, min(thresholds))
            
            # Find peaks of non-neutral states (the bifurcations)
            # peaks, _ = find_peaks(
            #     predictions[:-1],
            #     height=height,
            #     prominence=prominence
            # )
            
            # # Filter by distance
            # peaks = filter_points(
            #     points=peaks, 
            #     scores=predictions[peaks], 
            #     ages=ages, 
            #     min_distance=distance
            # )
        
        # values = f"Sequence Length: {seq_len}, Train Fraction: {train_fraction}, Epochs: {len(history.history['loss'])}/{epochs}, Training Loss: {history.history['loss'][-1]:.3f}, Valuation loss: {history.history['val_loss'][-1]:.3f}"

        def print_results():
            # print(f"Total sequences: {len(ages)}, trained on first {n_train} sequences with length {seq_len}")
            # print(values.replace(", ", "\n"))
            # for threshold in thresholds:
            #     print(f"Threshold: {threshold}th percentile -> {np.percentile(mse[:n_train], threshold):.4f}")

            # print(f"Detected {len(peaks)} tipping points:")
            # for point in peaks:
            #     print(f"Age {ages[point]:.0f}{age_format} with score {mse[point]:.2f}")
            pass
            
        def plot_results(parent):
                   
            # CNN-LSTM model plot results
                       
            main_axs: plt.Axes = parent.subplot(len(predictions), 2, sharex = True)
            for i in range(len(predictions)):
                axs: plt.Axes = main_axs[i, 1]
                axs.invert_xaxis()
                axs.set_title(f"CNN-LSTM predictions on {datasets[i].name}")
                p = predictions[i]
                axs.plot(datasets[i].ages(), p[:-1])
                axs.legend(["Fold", "Hopf", "Transcritical"])            
                axs.set_ylabel("Tipping probability")
                # for threshold in thresholds:
                #     axs.axhline(threshold, linestyle="--")
                # axs.scatter(ages[peaks], predictions[peaks], color="red", label="Tipping points")
                # for i, peak in enumerate(ages[peaks]):
                #     axs.annotate(peak, (peak, predictions[peaks][i]))

                # textaxs[1].text(0.5, 0.01, values, wrap=True, horizontalalignment='center', fontsize=12)
            
        return (print_results, plot_results)