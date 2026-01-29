#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:24:37 2026

Modified model training script for the tipping point-detecting deep learning model created by Thomas Bury

@author: Rhys Holloway, Thomas Bury

"""

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

##############################################

# TODO!

def run_with_args():
    
    import argparse
    parser = argparse.ArgumentParser(
                    prog='LSTM Model Trainer',
                    description='Generates training data')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()
    
if __name__ == "__main__":
    run_with_args()