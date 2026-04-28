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
from typing import Any, Callable, Final, Literal
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from .. import time_index, make_fast_lowess_residualizer
from . import INDEX_COL, Groups, Labels, TrainData, combine_batches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model, Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, Dense, LSTM, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report


_DEFAULT_EPOCHS = 1500

def train(
    data: TrainData,
    output: str,
    pad: Literal["lpad", "lrpad"],
    name: str = "best_model",
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = 1000,
    jobs: int = -1,
    verbose: bool = True,
) -> Sequential:
    sims, df_labels, df_groups = data
    labels = pd.merge(df_groups, df_labels, left_index=True, right_index=True)
    ts_len = len(next(iter(sims.values())))
    
    match pad:
        case "lrpad":
            pad_left = (ts_len // 2) - 25
            pad_right = pad_left
        case "lpad":
            pad_left = ts_len - 50
            pad_right = 0
        case _:
            raise ValueError("Please provide valid type as input: lrpad, lpad")
    
    residualize = make_fast_lowess_residualizer(np.arange(0, ts_len))
    
    def to_traindata(tsid: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        resids = residualize(sims[tsid])
        
        if len(resids) == 0:
            return resids 
        
        # Padding and normalizing input sequences
        left_padding = int(pad_left * random.random())
        right_padding = int(pad_right * random.random())
        if left_padding > 0:
            resids[:left_padding] = 0
        if right_padding > 0:
            resids[-right_padding:] = 0
        
        nonzero: np.ndarray[tuple[int], np.dtype[np.bool]] = resids != 0.0
        nz_resids = resids[nonzero]
        if not nonzero.any() or len(nz_resids) == 0:
            return resids
        
        avg = np.mean(np.abs(nz_resids))
        if not np.isfinite(avg) or avg == 0:
            return resids
        
        return resids / avg 
    
    os.makedirs(output, exist_ok=True)

    def format_name(stem: str, ext: str) -> str:
        return f"{stem}_{1 if pad == "lrpad" else 2}_len{ts_len}.{ext}"
    
    model_name = format_name(name, "keras")
    model_path = os.path.join(output, model_name)
    
    print("Computing training data from simulations...")

    print("Calculating", len(labels), "residuals")
    
    prepared: dict[int, np.ndarray] = dict(Parallel(
        n_jobs=jobs,
        backend="threading",
        prefer="threads",
        require="sharedmem",
        return_as="generator",
    )(
        delayed(to_traindata)(tsid)
        for tsid in labels.index.to_numpy(dtype=int)
    )) # type: ignore

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
    
    print("Loading or compiling model...")
    
    kernel_initializer: str = 'lecun_normal'
    
    def compile_model(model: Sequential):
        model.compile(
            loss=SparseCategoricalCrossentropy(),
            optimizer=Adam(learning_rate = 0.0005),
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )

    checkpoint_initial_value: float | None = None
    if os.path.exists(model_path):
        print(f"Loading existing model checkpoint from {model_path}...")
        model: Sequential = load_model(model_path) # type: ignore[assignment]

        input_shape = getattr(model, "input_shape", None)
        if input_shape is not None and tuple(input_shape[1:]) != (ts_len, 1):
            raise ValueError(
                f"Existing model at {model_path} expects input shape {input_shape[1:]}, "
                f"but training data has shape {(ts_len, 1)}"
            )

        if getattr(model, "optimizer", None) is None:
            print("Loaded model is not compiled; compiling before continuing training...")
            compile_model(model)

        validation_metrics = model.evaluate(
            validation,
            validation_target,
            batch_size=batch_size,
            verbose=0, # type: ignore
            return_dict=True,
        )
        checkpoint_initial_value = validation_metrics.get("accuracy")
        if checkpoint_initial_value is not None:
            print(f"Existing model validation accuracy: {checkpoint_initial_value}")
    else:
        print("No existing model checkpoint found; creating a new model...")
        model = Sequential([
            Input(shape=(ts_len, 1)),
            Conv1D(
                filters=50,
                kernel_size=12,
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
        compile_model(model)
    
    print("Fitting model...")
    
    history = model.fit(
        x=train,
        y=train_target,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1 if verbose else 0,
                initial_value_threshold=checkpoint_initial_value,
            ),
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
    
    
    # Kernel size: {model.layers[1].kernel_size}
    # Filters: {filters}
    # learning_rate: {optimizer.learning_rate}
    # kernel_initializer: {kernel_initializer}
    
    summary = f"""Simulation:
        Time Series Length: {ts_len}
        Epochs: {epochs}
        Batch size: {batch_size}
        
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
    
    def _read_batch(dir: str, jobs: int = 1) -> TrainData:
        print(f"Reading {dir}...")
        groups: pd.DataFrame = pd.read_csv(os.path.join(dir, "groups.csv"), index_col=INDEX_COL, memory_map=True)
        
        if len(groups) == 0:
            raise RuntimeError("Empty labels file!")
        
        print("Reading simulations of length", ts_len)   
        
        def read_sim(tsid: int) -> tuple[int, pd.Series]:    
            return tsid, pd.read_csv(os.path.join(dir, f"sims/tseries{tsid}.csv"), index_col=0, memory_map=True)['p0']     

        iterator: Any = Parallel(
            n_jobs=jobs,
            backend="threading",
            prefer="threads",
            require="sharedmem",
        )(
            delayed(read_sim)(int(tsid))
            for tsid in groups.index.values
        )

        sims: dict[int, pd.Series] = dict(iterator)
        
        labels = pd.read_csv(os.path.join(dir, "labels.csv"), index_col=INDEX_COL, memory_map=True)
        print("Read", dir)
        return sims, labels, groups
    
    if args.train is not None:
        print("Reading/Generating training data in", args.input)
        train_args = tuple(args.train)
        if len(train_args) not in [2, 3]:
            raise ValueError("Please provide the time series length (ex. 1500) AND the number of batches (ex. 10) OR an inclusive range of batches (ex. 1 10) and with --train! (ex. --train 1500 1 10 OR --train 1500 10)")
        import src.lstm.generate as generate
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
                batches: Any = Parallel(
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
        data=data,
        pad=args.pad,  
        name=args.name,
        output=output,
        epochs=args.epochs,
        jobs=args.jobs,
    )
