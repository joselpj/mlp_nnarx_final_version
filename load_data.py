import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import randint
import os
import datetime


def load_dataset(dataset_filename):
    df = pd.read_csv(os.path.join("data", dataset_filename), sep=",")
    dataset = df.iloc[:, :].values
    return dataset


def cascaded_tanks_dataset(na=1, nb=1, nk=1, normalize=True):
    dataset = load_dataset("cascaded_tanks.csv")
    col = na + nb
    row = dataset.shape[0] - nb

    x_train = np.zeros((row, col))
    y_train = np.zeros((row, 1))

    x_test = np.zeros((row, col))
    y_test = np.zeros((row, 1))

    for i in np.arange(na):
        x_train[:, i] = dataset[na - i:-(i + 1), 2]

    for j in np.arange(na, col):
        x_train[:, j] = dataset[col - 1 - j:-nb + (col - 1 - j), 0]

    y_train[:, 0] = dataset[na + 1:, 2]

    for i in np.arange(na):
        x_test[:, i] = dataset[na - i:-(i + 1), 3]

    for j in np.arange(na, col):
        x_test[:, j] = dataset[col - 1 - j:-nb + (col - 1 - j), 1]

    y_test[:, 0] = dataset[na + 1:, 3]

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)
        x_test = scaler.fit_transform(x_test)
        y_test = scaler.fit_transform(y_test)

    return x_train, y_train, x_test, y_test


def gas_furnace_dataset(na=1, nb=1, nk=1, normalize=True):
    dataset = load_dataset("gas-furnace.csv")
    dataset_train = dataset[:-40, :]
    dataset_test = dataset[dataset_train.shape[0]:, :]
    col = na + nb
    row_train = dataset_train.shape[0] - nb
    row_test = dataset_test.shape[0] - nb

    x_train = np.zeros((row_train, col))
    y_train = np.zeros((row_train, 1))

    x_test = np.zeros((row_test, col))
    y_test = np.zeros((row_test, 1))

    for i in np.arange(na):
        x_train[:, i] = dataset_train[na - i:-(i + 1), 1]

    for j in np.arange(na, col):
        x_train[:, j] = dataset_train[col - 1 - j:-nb + (col - 1 - j), 0]

    y_train[:, 0] = dataset_train[na + 1:, 1]

    for i in np.arange(na):
        x_test[:, i] = dataset_test[na - i:-(i + 1), 1]

    for j in np.arange(na, col):
        x_test[:, j] = dataset_test[col - 1 - j:-nb + (col - 1 - j), 0]

    y_test[:, 0] = dataset_test[na + 1:, 1]

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)

        x_test = scaler.fit_transform(x_train)
        y_test = scaler.fit_transform(y_train)

    return x_train, y_train, x_test, y_test


def silver_box_dataset(na=1, nb=1, nk=1, normalize=True):
    # v1: input
    # v2 output
    dataset = load_dataset("Schroeder80mV.csv")
    dataset_train = dataset[0:1001, :]
    dataset_test = dataset[1001:2001, :]
    col = na + nb
    row_train = dataset_train.shape[0] - nb
    row_test = dataset_test.shape[0] - nb

    x_train = np.zeros((row_train, col))
    y_train = np.zeros((row_train, 1))

    x_test = np.zeros((row_test, col))
    y_test = np.zeros((row_test, 1))

    for i in np.arange(na):
        x_train[:, i] = dataset_train[na - i:-(i + 1), 3]

    for j in np.arange(na, col):
        x_train[:, j] = dataset_train[col - 1 - j:-nb + (col - 1 - j), 2]

    y_train[:, 0] = dataset_train[na + 1:, 3]

    for i in np.arange(na):
        x_test[:, i] = dataset_test[na - i:-(i + 1), 3]

    for j in np.arange(na, col):
        x_test[:, j] = dataset_test[col - 1 - j:-nb + (col - 1 - j), 2]

    y_test[:, 0] = dataset_test[na + 1:, 3]

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)

        x_test = scaler.fit_transform(x_train)
        y_test = scaler.fit_transform(y_train)

    return x_train, y_train, x_test, y_test


def wiener_hammer_dataset(na=1, nb=1, nk=1, normalize=True):
    dataset = load_dataset("WienerHammerBenchmark.csv")
    dataset_train = dataset[5000:6001, :] # [0:1001, :]
    dataset_test = dataset[6001:7001, :] # [0:1001, :]
    col = na + nb
    row_train = dataset_train.shape[0] - nb
    row_test = dataset_test.shape[0] - nb

    x_train = np.zeros((row_train, col))
    y_train = np.zeros((row_train, 1))

    x_test = np.zeros((row_test, col))
    y_test = np.zeros((row_test, 1))

    for i in np.arange(na):
        x_train[:, i] = dataset_train[na - i:-(i + 1), 1]

    for j in np.arange(na, col):
        x_train[:, j] = dataset_train[col - 1 - j:-nb + (col - 1 - j), 0]

    y_train[:, 0] = dataset_train[na + 1:, 1]

    for i in np.arange(na):
        x_test[:, i] = dataset_test[na - i:-(i + 1), 1]

    for j in np.arange(na, col):
        x_test[:, j] = dataset_test[col - 1 - j:-nb + (col - 1 - j), 0]

    y_test[:, 0] = dataset_test[na + 1:, 1]

    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(x_train)
        y_train = scaler.fit_transform(y_train)

        x_test = scaler.fit_transform(x_train)
        y_test = scaler.fit_transform(y_train)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = cascaded_tanks_dataset(4, 5, 1, normalize=False)
    print("x_train: \n", x_train[0, :])
    print("\n y_train: \n", y_train[0])
    print("x_test: \n", x_test[0, :])
    print("\n y_test: \n", y_test[0])
    x_train, y_train, x_test, y_test = gas_furnace_dataset(4, 5, 1, normalize=False)
    print("x_train: \n", x_train[0, :])
    print("\n y_train: \n", y_train[0])
    print("x_test: \n", x_test[0, :])
    print("\n y_test: \n", y_test[0])

    x_train, y_train, x_test, y_test = silver_box_dataset(4, 5, 1, normalize=False)
    print("x_train: \n", x_train[0, :])
    print("\n y_train: \n", y_train[0])
    print("x_test: \n", x_test[0, :])
    print("\n y_test: \n", y_test[0])

    x_train, y_train, x_test, y_test = wiener_hammer_dataset(4, 5, 1, normalize=False)
    print("x_train: \n", x_train[0, :])
    print("\n y_train: \n", y_train[0])
    print("x_test: \n", x_test[0, :])
    print("\n y_test: \n", y_test[0])


