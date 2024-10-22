import argparse
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def _init_argparse():
    parser = argparse.ArgumentParser(description="Process some CSV data.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join(os.getcwd(), "data/hourly_data_features.csv"),
        help="Path to the CSV file (default: current directory)",
    )
    return parser


def preprocess_data(df, target_column="Target", test_size=0.15, val_size=0.15):
    # Fix Volume Change Column
    df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], np.nan)
    df['Volume_Change'] = df['Volume_Change'].fillna(0)

    # TODO - One hot encoded the Close_Position_Category column
    X = df.drop(columns=[target_column, "Ticker", "Date", "Close_Position_Category"])
    y = df[target_column]

    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42
    )

    # Standardize the data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)

    return (
        (X_train_tensor, y_train_tensor),
        (X_val_tensor, y_val_tensor),
        (X_test_tensor, y_test_tensor),
        scaler,
    )


def run(file_path):
    # Load & Pre-Process Data
    df = pd.read_csv(file_path)
    # print(df.describe())
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = preprocess_data(df)

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")


if __name__ == "__main__":
    parser = _init_argparse()
    args = parser.parse_args()

    run(args.file_path)
