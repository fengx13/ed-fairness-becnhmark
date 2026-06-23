"""Keras sequence generator for LSTM models over ED vital-sign time series.

Isolated here so that importing the rest of :mod:`fair_ed` does not require
TensorFlow to be installed.
"""
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


class LSTMDataGenerator(Sequence):
    def __init__(self, main_df, vitalsign_df, y, batch_size, x1_cols, x2_cols):
        self.main_df = main_df
        self.vitalsign_df = vitalsign_df
        self.batch_size = batch_size
        self.x1_cols = x1_cols
        self.x2_cols = x2_cols
        self.y_df = y

    def __len__(self):
        return math.ceil(len(self.main_df) / self.batch_size)

    def __getitem__(self, index):
        df_batch = self.main_df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        x1 = df_batch[self.x1_cols].to_numpy().astype(np.float64)
        y = self.y_df.iloc[index * self.batch_size:(index + 1) * self.batch_size].to_numpy()
        stay_ids = df_batch['stay_id'].to_numpy().astype(np.int64)
        batch_size = len(df_batch)
        df_batch = df_batch.merge(self.vitalsign_df, on='stay_id', how='left')
        x2 = []
        for i in range(batch_size):
            x2.append(df_batch[df_batch['stay_id'] == stay_ids[i]][self.x2_cols].to_numpy())
        padded_x2 = pad_sequences(x2, padding='post')
        return [x1, padded_x2.astype(np.float64)], y


def get_lstm_data_gen(df_train, df_test, df_vitalsign, variable, outcome, batch_size=200):
    variable_with_id = ["stay_id"]
    variable_with_id.extend(variable)

    X_train = df_train[variable_with_id].copy()
    y_train = df_train[outcome].copy()
    X_test = df_test[variable_with_id].copy()
    y_test = df_test[outcome].copy()

    if 'gender' in variable:
        encoder = LabelEncoder()
        X_train['gender'] = encoder.fit_transform(X_train['gender'])
        X_test['gender'] = encoder.transform(X_test['gender'])

    if 'ed_los' in variable:
        X_train['ed_los'] = pd.to_timedelta(X_train['ed_los']).dt.seconds / 60
        X_test['ed_los'] = pd.to_timedelta(X_test['ed_los']).dt.seconds / 60

    x1_cols = [x for x in variable_with_id[1:] if not ('ed' in x and 'last' in x)]
    x2_cols = [x for x in df_vitalsign.columns if 'ed' in x]

    train_data_gen = LSTMDataGenerator(X_train, df_vitalsign, y_train, batch_size, x1_cols, x2_cols)
    test_data_gen = LSTMDataGenerator(X_test, df_vitalsign, y_test, batch_size, x1_cols, x2_cols)

    return train_data_gen, test_data_gen
