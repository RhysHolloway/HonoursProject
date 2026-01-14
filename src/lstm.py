import os
from typing import Any, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
from model import filter_points

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Conv1D, Dropout, MaxPooling1D, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping    
import tensorflow as tf

def __fixed_length_sequence(features, seq_len) -> tuple[np.array, int]:
            
    # === Scale features ===
    features_scaled = StandardScaler().fit_transform(features)

    # === Make sequences ===
    if len(features_scaled) < seq_len + 10:
        raise ValueError(
            f"Time series too short for seq_len={seq_len}. "
            f"Need at least seq_len+10 points."
        )
    
    X = []
    for i in range(len(features_scaled) - seq_len + 1):
        X.append(features_scaled[i:i + seq_len])
    return (np.array(X), seq_len)  # (n_seq, seq_len, n_features)

def fixed_length_sequence(length: int):    
    return lambda features: __fixed_length_sequence(features, length)
        
def lstm_tipping_points(
    sequencer: Callable[[Any, int], tuple[np.array, int]],
    train_fraction: float,
    epochs: int,
    patience: int = 30,
    batch_size: int = 32,
    latent_dim: int = 16,
    kernel_size: int = 12,
    conv_filters: int = 50,
    threshold: Union[Union[int, float], list[Union[int, float]]] = [95, 99],
    prominence: float = 0.05,
    smoothing_window: int = 1,
    distance: int = 10,
):
    
    def __lstm_tipping_points(
        ages: Any,
        features: Any,
        thresholds=threshold if isinstance(threshold, list) else [threshold],
    ):
        
        X_all, seq_len = sequencer(features) 
        seq_ages = ages[seq_len - 1:]

        n_train = int(train_fraction * len(X_all))
        X_train = X_all[:n_train]

        print(f"Total sequences: {len(X_all)}, training on first {n_train} sequences with length {seq_len}")
        
        inputs = Input(shape=(seq_len, features.shape[1]))
        layers = Conv1D(filters=conv_filters, kernel_size=kernel_size,padding="same",activation="relu")(inputs)
        layers = Dropout(0.1)(layers)
        layers = MaxPooling1D()(layers)
        layers = LSTM(latent_dim)(layers)
        layers = RepeatVector(seq_len)(layers)
        layers = LSTM(latent_dim, return_sequences=True, dropout=0.1)(layers)
        # layers = Dropout(0.1)(layers)
        layers = TimeDistributed(Dense(features.shape[1]))(layers)

        model = Model(inputs, layers)
        model.compile(optimizer="adam", loss="mse")

        # if verbose:
        #     model.summary()

        history = model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
            verbose=False,
        )

        # === Reconstruction error for all sequences ===
        X_pred = model.predict(
            X_all, 
            verbose=False
        )
        # mse over time and features
        mse = np.mean((X_all - X_pred) ** 2, axis=(1, 2))  # shape (n_seq,)

        # === Smooth the error to get a tipping-point score ===
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            mse = np.convolve(mse, kernel, mode="same")

        height = np.percentile(mse[:n_train], min(thresholds))

        
        peaks, _ = find_peaks(
            mse,
            height=height,
            prominence=prominence
        )
        
        peaks = filter_points(
            points=peaks, 
            scores=mse[peaks], 
            ages=seq_ages, 
            min_distance=distance
        )
        
        values = f"Sequence Length: {seq_len}, Train Fraction: {train_fraction}, Epochs: {len(history.history['loss'])}/{epochs}, Training Loss: {history.history['loss'][-1]:.3f}, Valuation loss: {history.history['val_loss'][-1]:.3f}"

        def print_results(age_format: str):
            print(values.replace(", ", "\n"))
            for threshold in thresholds:
                print(f"Threshold: {threshold}th percentile -> {np.percentile(mse[:n_train], threshold):.4f}")

            print(f"Detected {len(peaks)} tipping points:")
            for point in peaks:
                print(f"Age {seq_ages[point]:.0f}{age_format} with score {mse[point]:.2f}")
            
        def plot_results(age_format: str):
            plt.plot(seq_ages, mse)
            plt.ylabel(f"Tipping score")
            plt.figtext(0.5, 0.01, values, wrap=True, horizontalalignment='center', fontsize=12)
            for threshold in thresholds:
                plt.axhline(np.percentile(mse[:n_train], threshold), linestyle="--", label=f"{threshold}th percentile threshold")

            plt.scatter(seq_ages[peaks], mse[peaks], color="red", label="Tipping points")
            for i, peak in enumerate(seq_ages[peaks]):
                plt.annotate(peak, (peak, mse[peaks][i]))
        
        return (print_results, plot_results)
    
    return (
        "LSTM", 
        lambda ages, features : __lstm_tipping_points(
            ages,
            features
        ),
    )