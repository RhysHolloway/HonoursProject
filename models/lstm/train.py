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
from typing import Final, Literal, Union
import numpy as np
import pandas as pd

from ..lstm import INDEX_COL, TrainData, combine_batches, compute_residuals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.models import Sequential  # type: ignore
from keras.layers import Dropout, Conv1D, MaxPooling1D, Dense, LSTM, Input  # type: ignore
from keras.optimizers import Optimizer, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report

def _read_batch(dir) -> tuple[TrainData, int]:
    print(f"Reading {dir}...")
    groups = pd.read_csv(os.path.join(dir, "groups.csv"), index_col=INDEX_COL) 
    
    if len(groups) == 0:
        raise RuntimeError("Empty labels file!")

    sims = {seq_id:pd.read_csv(os.path.join(dir, f"sims/tseries{seq_id}.csv")) for seq_id in groups.index.values}
    
    ts_len = len(next(iter(sims.values()))[0])
    
    print("Read", dir)
    
    return (sims, pd.read_csv(os.path.join(dir, "labels.csv"), index_col=INDEX_COL), groups), ts_len

def _read_input_folder(input: str) -> tuple[TrainData, int]:
    try:
        entries = os.listdir(input)
        if "groups.csv" in entries:
            batches, ts_len = _read_batch(input)
        else:
            batches = [entry for entry in entries if os.path.exists(os.path.join(input, entry, "groups.csv"))]
            if len(batches) == 0:
                raise RuntimeError(f"Input folder {input} contains no batches in entries: {entries}")
            batches = [_read_batch(os.path.join(input, entry)) for entry in entries] 
            ts_len = batches[0][-1]
            if not all(ts_len == other_len for _, other_len in batches):
                raise RuntimeError(f"Input series lengths are different! {list(other_len for _, other_len in batches)}")
            batches = combine_batches(b[0] for b in batches)
            
    except Exception as e:
        raise RuntimeError("Could not read input folder with error:", e) 
    
    return batches, ts_len

def train_lstm_from_batches(
    input: str,
    output: Union[str, None],
    type: Literal["lpad", "lrpad"],
    epochs: int,
    name: str = "best_model",
    patience: int = 30,
    batch_size: int = 32,
    filters: int = 50,
    kernel_size: int = 12,
    kernel_initializer: str = 'lecun_normal',
    optimizer: Optimizer = Adam(learning_rate = 0.0005),
) -> Sequential:
    output = input if output is None else output
    
    batches, ts_len = _read_input_folder(input)
                
    sims, df_targets, df_groups = batches
    
    print("Setting up training data...")
    
    match type:
        case "lrpad":
            pad_left = (ts_len // 2) - 25
            pad_right = pad_left
        case "lpad":
            pad_left = ts_len - 50
            pad_right = 0

    def transform(sim: pd.Series) -> np.ndarray:
        values = compute_residuals(sim).to_numpy(copy=True)
        
        # Padding and normalizing input sequences
        values[:int(pad_left * random.uniform(0, 1))] = 0
        values[len(values)-int(pad_right * random.uniform(0, 1)):] = 0
        
        avg = sum(np.abs(values)) / np.count_nonzero(values)
        
        return values / avg

    # apply train/test/validation labels
    
    labelled_seq = lambda label: np.array([transform(sims[tsid][1]) for tsid in df_groups[df_groups["dataset_ID"] == label].index])
    
    train = labelled_seq(1)
    validation = labelled_seq(2)
    test = labelled_seq(3)
    
    labelled_class_seq = lambda label: df_targets[df_groups["dataset_ID"] == label]["class_label"].to_numpy()
    
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
        Dense(4, activation='softmax', kernel_initializer = kernel_initializer)
    ])

    model.compile(
        loss=SparseCategoricalCrossentropy(), 
        optimizer=optimizer, 
        metrics=['accuracy']
    )
    
    print("Fitting model...")
    

    def format_name(object: str, ext: str):
        return f"{object}_{1 if type == "lrpad" else 2}_len{ts_len}.{ext}"
    
    model_name = format_name(name, "keras")
    MONITOR: Final[str] = "val_accuracy"
    
    history = model.fit(
        x=train,
        y=train_target,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(monitor=MONITOR, patience=patience, restore_best_weights=True), 
            ModelCheckpoint(model_name, monitor=MONITOR, save_best_only=True, mode="max", verbose=1)
        ],
        validation_data=(validation, validation_target),
        verbose=True,
    )
    
    print("Outputting model...")
    
    os.makedirs(output, exist_ok=True)
    model.save(os.path.join(output, model_name))
    
    print("Testing model...")
    
    # generate test metrics

    test_preds = model.predict(test).argmax(axis=-1)
    accuracy_score(test_target, test_preds)
    
    output = f"""Simulation: 
        Time Series Length: {ts_len}
        Epochs: {epochs}
        Kernel size: {kernel_size}
        Filters: {filters}
        Batch size: {batch_size}
        learning_rate: {optimizer.learning_rate}
        kernel_initializer: {kernel_initializer}
        pad_left: {pad_left}
        pad_right: {pad_right}
        
        macro f1: {f1_score(test_target, test_preds, average="macro")}
        macro avg precision: {precision_score(test_target, test_preds, average="macro")}
        macro avg recall: {recall_score(test_target, test_preds, average="macro")}
        """
    print(output)
    print(classification_report(test_target, test_preds, digits=3))
    print(history.history["accuracy"])
    print(history.history["val_accuracy"])
    print(history.history["loss"])
    print(history.history["val_loss"])
    print("Confusion matrix: \n", confusion_matrix(test_target, test_preds))# keeps track of training metrics

    with open(name("training_results", "txt"), "w") as results:
        results.write(output)
        results.flush()
        
    return model


#############################################

def run_with_args():
    
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Trainer',
                    description='Generates training data')
    parser.add_argument('input', type=str, help="Path to a folder containing batches or a generated batch")
    parser.add_argument('--output', '-o', type=str, help="Folder path to output models. If not provided, defaults to placing model beside training data.", default=None)
    parser.add_argument('--name', '-n', type=str, help="Model name", default="best_model")
    parser.add_argument('--epochs', '-e', type=int, help="Number of epochs to train the model for", default=500)
    parser.add_argument('--patience', '-p', type=int, help="Cancel training after a given number of epochs if the model does not improve since then", default=50)
    parser.add_argument('--type', '-t', type=str, help="Type of zero-padding to use training the model", default="lrpad")
    args = parser.parse_args()
    
    train_lstm_from_batches(args.input, args.output, type=args.type, epochs=args.epochs, patience=args.patience)
    
if __name__ == "__main__":
    run_with_args()