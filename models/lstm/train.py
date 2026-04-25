#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:24:37 2026

Modified model training script for the tipping point-detecting deep learning model created by Thomas Bury

@author: Rhys Holloway, Thomas Bury

"""

import os
import os.path
import random
from typing import Callable, Final, Literal
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from .. import compute_residuals, index_values
from ..lstm import INDEX_COL, TrainData, combine_batches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.models import load_model, Sequential  # type: ignore
from keras.layers import Dropout, Conv1D, MaxPooling1D, Dense, LSTM, Input  # type: ignore
from keras.optimizers import Optimizer, Adam
from keras.callbacks import ModelCheckpoint  # type: ignore
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report

def make_fast_lowess_residualizer(index: pd.Index, span: float = 0.2) -> Callable[[pd.Series], np.ndarray]:
    from scipy import sparse

    x = index_values(index)
    n = len(x)
    if n == 0:
        return lambda data: np.empty(0, dtype=float)

    frac = span if 0 < span <= 1 else span / n
    window = min(n, max(2, int(np.ceil(frac * n))))

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []

    for row, x0 in enumerate(x):
        distances = np.abs(x - x0)
        neighbour_idx = np.argpartition(distances, window - 1)[:window]
        h = distances[neighbour_idx].max()
        if h == 0:
            rows.append(row)
            cols.append(row)
            weights.append(1.0)
            continue

        xdiff = x[neighbour_idx] - x0
        local_weights = (1 - (np.abs(xdiff) / h) ** 3) ** 3
        valid = local_weights > 0
        neighbour_idx = neighbour_idx[valid]
        xdiff = xdiff[valid]
        local_weights = local_weights[valid]

        s0 = local_weights.sum()
        s1 = np.dot(local_weights, xdiff)
        s2 = np.dot(local_weights, xdiff * xdiff)
        denom = s0 * s2 - s1 * s1

        if denom == 0:
            row_weights = local_weights / s0
        else:
            row_weights = local_weights * (s2 - s1 * xdiff) / denom

        rows.extend([row] * len(neighbour_idx))
        cols.extend(neighbour_idx.tolist())
        weights.extend(row_weights.tolist())

    smoother = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=float)

    def residualize(data: pd.Series) -> np.ndarray:
        state = data.to_numpy(dtype=float, copy=False)
        if len(state) != n:
            raise ValueError(f"FastLowess residualizer expected length {n}, got {len(state)}")
        if not np.isfinite(state).all():
            return compute_residuals(data, span=span, type="Lowess").to_numpy(dtype=float, copy=True)
        return state - smoother.dot(state)

    return residualize


_DEFAULT_EPOCHS = 500
_DEFAULT_PATIENCE = 50
def train(
    data: TrainData,
    output: str,
    pad: Literal["lpad", "lrpad"],
    name: str = "best_model",
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = 1000,
    filters: int = 50,
    kernel_size: int = 12,
    kernel_initializer: str = 'lecun_normal',
    optimizer: Optimizer | None = None,
    jobs: int = -1,
    verbose: bool = True,
) -> Sequential:
    sims, df_labels, df_groups = data
    labels = pd.merge(df_groups, df_labels, left_index=True, right_index=True)
    ts_len = len(next(iter(sims.values())))
    optimizer = optimizer or Adam(learning_rate = 0.0005)
        
    match pad:
        case "lrpad":
            pad_left = (ts_len // 2) - 25
            pad_right = pad_left
        case "lpad":
            pad_left = ts_len - 50
            pad_right = 0
        case _:
            raise ValueError("Please provide valid type as input: lrpad, lpad")
    
    os.makedirs(output, exist_ok=True)

    def format_name(stem: str, ext: str) -> str:
        pad_id = 1 if pad == "lrpad" else 2
        return f"{stem}_{pad_id}_len{ts_len}.{ext}"
    
    model_name = format_name(name, "keras")
    model_path = os.path.join(output, model_name)
    
    print("Computing training data from simulations...")

    residualize = make_fast_lowess_residualizer(next(iter(sims.values())).index)
    
    def to_traindata(tsid: int) -> tuple[int, np.ndarray]:
        values = np.array(residualize(sims[tsid]), dtype=float, copy=True)
        
        # Padding and normalizing input sequences
        left_padding = int(pad_left * random.random())
        right_padding = int(pad_right * random.random())
        if left_padding > 0:
            values[:left_padding] = 0
        if right_padding > 0:
            values[-right_padding:] = 0
        
        nonzero = values != 0
        avg = np.mean(np.abs(values[nonzero])) if np.any(nonzero) else 1.0
        if not np.isfinite(avg) or avg == 0:
            avg = 1.0
        
        return tsid, (values / avg).astype(np.float32, copy=False)

    sequence_ids = labels.index.to_numpy(dtype=int)
    total_sequences = len(sequence_ids)

    prepared_results = Parallel(
        n_jobs=jobs,
        backend="threading",
        prefer="threads",
        require="sharedmem",
        return_as="generator",
    )(
        delayed(to_traindata)(tsid)
        for tsid in sequence_ids
    )
    
    print("Calculating", total_sequences, "residuals")
    
    
    prepared: dict[int, np.ndarray] = {tsid:values for tsid, values in prepared_results} # type: ignore

    def sequence_ids_for(dataset_id: int) -> np.ndarray:
        return labels.index[labels["dataset_ID"] == dataset_id].to_numpy(dtype=int)

    def labelled_seq(dataset_id: int) -> np.ndarray:
        ids = sequence_ids_for(dataset_id)
        if len(ids) == 0:
            return np.empty((0, ts_len, 1), dtype=np.float32)
        return np.stack([prepared[tsid] for tsid in ids]).reshape(len(ids), ts_len, 1)
    
    train = labelled_seq(1)
    validation = labelled_seq(2)
    test = labelled_seq(3)
    
    def labelled_class_seq(dataset_id: int) -> np.ndarray:
        return labels.loc[sequence_ids_for(dataset_id), "class_label"].to_numpy(dtype=np.int64)
    
    train_target = labelled_class_seq(1)
    validation_target = labelled_class_seq(2)
    test_target = labelled_class_seq(3)
    
    print("Compiling model...")
    
    model = Sequential([
        Input(shape=(ts_len, 1)),
        Conv1D(
            filters=filters, 
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_initializer=kernel_initializer,
        ),
        Dropout(0.1),
        MaxPooling1D(pool_size = 2),
        LSTM(50, return_sequences=True, kernel_initializer = kernel_initializer),
        Dropout(0.1),
        LSTM(10, kernel_initializer = kernel_initializer),
        Dropout(0.1),
        Dense(4, activation='softmax', kernel_initializer = kernel_initializer)
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(), 
        optimizer=optimizer, 
        metrics=['accuracy', 'sparse_categorical_accuracy']
    )
    
    print("Fitting model...")
    
    MONITOR: Final[str] = "val_accuracy"
    
    history = model.fit(
        x=train,
        y=train_target,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(model_path, monitor=MONITOR, save_best_only=True, mode="max", verbose=1 if verbose else 0),
        ],
        validation_data=(validation, validation_target),
        verbose="auto" if verbose else 0, # type: ignore
    )
    
    print("Loading best model checkpoint...")

    model: Sequential = load_model(model_path) # type: ignore[assignment]
    
    print("Testing model...")
    
    # generate test metrics

    test_preds = model.predict(test, batch_size=256).argmax(axis=-1)
    accuracy = accuracy_score(test_target, test_preds)
    
    summary = f"""Simulation:
        Time Series Length: {ts_len}
        Epochs: {epochs}
        Kernel size: {kernel_size}
        Filters: {filters}
        Batch size: {batch_size}
        learning_rate: {optimizer.learning_rate}
        kernel_initializer: {kernel_initializer}
        pad_left: {pad_left}
        pad_right: {pad_right}
        
        accuracy: {accuracy}
        macro f1: {f1_score(test_target, test_preds, average="macro")}
        macro avg precision: {precision_score(test_target, test_preds, average="macro")}
        macro avg recall: {recall_score(test_target, test_preds, average="macro")}
        
        last accuracy: {history.history["accuracy"][-1]}
        validation accuracy: {history.history["val_accuracy"][-1]}
        loss: {history.history["loss"][-1]}
        validation loss: {history.history["val_loss"][-1]}
        """
    print(summary)
    print(classification_report(test_target, test_preds, digits=3))
    print(history.history["accuracy"])
    print(history.history["val_accuracy"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    print("Confusion matrix: \n", confusion_matrix(test_target, test_preds))# keeps track of training metrics

    with open(os.path.join(output, format_name("training_results", "txt")), "w") as results:
        results.write(summary)
        results.flush()
        
    return model


#############################################

if __name__ == "__main__":
    
    def _read_batch(dir: str, jobs: int = 1) -> TrainData:
        print(f"Reading {dir}...")
        groups: pd.DataFrame = pd.read_csv(os.path.join(dir, "groups.csv"), index_col=INDEX_COL)
        
        if len(groups) == 0:
            raise RuntimeError("Empty labels file!")

        def read_sim(seq_id: int) -> tuple[int, pd.Series]:
            return seq_id, pd.read_csv(os.path.join(dir, f"sims/tseries{seq_id}.csv"), index_col=0)['p0']

        sims = dict(Parallel(
            n_jobs=jobs,
            backend="threading",
            prefer="threads",
            require="sharedmem",
        )(
            delayed(read_sim)(int(seq_id))
            for seq_id in groups.index.values
        ))
        labels = pd.read_csv(os.path.join(dir, "labels.csv"), index_col=INDEX_COL)
        print("Read", dir)
        return sims, labels, groups
    
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Trainer',
                    description='Trains models on generated data')
    parser.add_argument('input', type=str, help="Path to a folder containing batches or a generated batch")
    parser.add_argument('--train', '-t', type=int, nargs="+", help="Generate the set of training data before training the model.")
    parser.add_argument('--pad', '-p', type=str, help="Type of zero-padding to use training the model", default="lrpad")
    parser.add_argument('--name', '-n', type=str, help="Model file name", default="best_model")
    parser.add_argument('--output', '-o', type=str, help="Folder path to output models. If not provided, defaults to placing model beside training data.")
    parser.add_argument('--epochs', '-e', type=int, help="Number of epochs to train the model for", default=_DEFAULT_EPOCHS)
    parser.add_argument('--jobs', '-j', type=int, help="Number of joblib worker threads to use for data preparation.", default=-1)
    
    args = parser.parse_args()
    
    output = args.input if args.output is None else args.output
    
    if args.train is not None:
        print("Reading/Generating training data in", args.input)
        train_args = tuple(args.train)
        if len(train_args) not in [2, 3]:
            raise ValueError("Please provide the time series length (ex. 1500) AND the number of batches (ex. 10) OR an inclusive range of batches (ex. 1 10) and with --train! (ex. --train 1500 1 10 OR --train 1500 10)")
        import models.lstm.generate as generate
        data = generate.run_with_args(
            ts_len=train_args[0], 
            batches=list(range(train_args[1] if len(train_args) == 3 else 1, train_args[-1] + 1)),
            output=args.input
        )
    else:
        input = args.input
        try:
            entries = os.listdir(input)
            if "groups.csv" in entries:
                data = _read_batch(input, jobs=args.jobs)
            else:
                batch_dirs = sorted(entry for entry in entries if os.path.exists(os.path.join(input, entry, "groups.csv")))
                if len(batch_dirs) == 0:
                    raise RuntimeError(f"Input folder {input} contains no batches in entries: {entries}")
                batches = Parallel(
                    n_jobs=args.jobs,
                    backend="threading",
                    prefer="threads",
                    require="sharedmem",
                )(
                    delayed(_read_batch)(os.path.join(input, entry), 1)
                    for entry in batch_dirs
                )
                ts_len = len(next(iter(batches[0][0].values())))
                if not all(ts_len == len(next(iter(b[0].values()))) for b in batches):
                    raise RuntimeError(f"Input series lengths are different!")
                data = combine_batches(batches)
                
        except Exception as e:
            raise RuntimeError("Could not read input folder with error:", e) 
    
    train(
        data=data,  # type: ignore
        pad=args.pad,  
        name=args.name,
        output=output,
        epochs=args.epochs,
        jobs=args.jobs,
    )
