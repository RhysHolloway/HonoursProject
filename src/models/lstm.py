from typing import Any, Callable, Union


import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from util import filter_points
from tensorflow.keras.models import load_model
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
    
#     return (model, history)

model = None

# Make sure to do linear interpolation before using
def lstm(
    # model,
    threshold: Union[float, list[float]] = [0.95, 0.99],
    prominence: float = 0.05,
    distance: int = 10,
):
    
    def run(
        ages: Any,
        features: Any,
        thresholds: float = threshold if isinstance(threshold, list) else [threshold],
    ):
        
        if model is None:
            model = load_model(get_project_path("env/deep-early-warnings-pnas/dl_train/best_models_tf215/len500/best_model_1_1_len500.keras"))
    
        seq_features = np.array(features) / np.mean(np.abs(features))

        predictions = model.predict(seq_features, verbose=False)

        height = np.percentile(predictions, min(thresholds))
        
        # Find peaks of non-neutral states (the bifurcations)
        peaks, _ = find_peaks(
            predictions[:-1],
            height=height,
            prominence=prominence
        )
        
        # Filter by distance
        peaks = filter_points(
            points=peaks, 
            scores=predictions[peaks], 
            ages=ages, 
            min_distance=distance
        )
        
        values = f"Sequence Length: {seq_len}, Train Fraction: {train_fraction}, Epochs: {len(history.history['loss'])}/{epochs}, Training Loss: {history.history['loss'][-1]:.3f}, Valuation loss: {history.history['val_loss'][-1]:.3f}"

        def print_results(_data_name: str, age_format: str):
            print(f"Total sequences: {len(ages)}, trained on first {n_train} sequences with length {seq_len}")
            print(values.replace(", ", "\n"))
            for threshold in thresholds:
                print(f"Threshold: {threshold}th percentile -> {np.percentile(mse[:n_train], threshold):.4f}")

            print(f"Detected {len(peaks)} tipping points:")
            for point in peaks:
                print(f"Age {ages[point]:.0f}{age_format} with score {mse[point]:.2f}")
            
        def plot_results(parent, data_name: str, age_format: str):
                   
            # CNN-LSTM model plot results
                       
            axs = parent.subplot(1, 2, figsize=(8,5), sharex = True)
            axs, textaxs = axs[1, 1], axs[1, 2]
            axs.invert_xaxis()
            fig.suptitle("CNN-LSTM predictions")
            axs.plot(ages, predictions[:-1])
            fig.legend(["Fold", "Hopf", "Transcritical"])            
            fig.ylabel("Tipping probability")
            for threshold in thresholds:
                axs.axhline(threshold, linestyle="--")
            axs.scatter(ages[peaks], predictions[peaks], color="red", label="Tipping points")
            for i, peak in enumerate(ages[peaks]):
                axs.annotate(peak, (peak, predictions[peaks][i]))

            textaxs[1].text(0.5, 0.01, values, wrap=True, horizontalalignment='center', fontsize=12)
            
        return (print_results, plot_results)
    
    return ("LSTM", run)