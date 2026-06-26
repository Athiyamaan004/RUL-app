import os
import glob
import pickle

import numpy as np
import pandas as pd

import torch

from model import CNN_LSTM
from feature_extraction import extract_vibration_features

import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


FEATURE_COUNT = 21

WIN_LEN = 10


model = CNN_LSTM(
    input_features=FEATURE_COUNT
)

model.load_state_dict(
    torch.load(
        resource_path("best_model.pth"),
        map_location="cpu"
    )
)

model.eval()


with open(
    resource_path("scaler.pkl"),
    "rb"
) as f:

    scaler = pickle.load(f)


features = [

    "damage_index",

    "health_trend",

    "combined_health",

    "health_index_h",
    "health_index_v",

    "rolling_rms_h",
    "rolling_rms_v",

    "rolling_std_h",
    "rolling_std_v",

    "rms_h",
    "rms_v",

    "std_h",
    "std_v",

    "var_h",
    "var_v",

    "spec_energy_h",
    "spec_energy_v",

    "mean_fft_h",
    "mean_fft_v",

    "spectral_centroid_h",
    "spectral_centroid_v"
]
def predict_bearing(folder_path):

    acc_files = sorted(

        glob.glob(

            os.path.join(
                folder_path,
                "acc_*.csv"
            )

        )

    )

    if len(acc_files) == 0:

        raise Exception(
            "No vibration CSV files found."
        )

    rows = []

    for file in acc_files:

        rows.append(

            extract_vibration_features(file)

        )

    feature_df = pd.DataFrame(rows)

    feature_df["rolling_rms_h"] = (
        feature_df["rms_h"]
        .rolling(5, min_periods=1)
        .mean()
    )

    feature_df["rolling_rms_v"] = (
        feature_df["rms_v"]
        .rolling(5, min_periods=1)
        .mean()
    )

    feature_df["rolling_std_h"] = (
        feature_df["rms_h"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0)
    )

    feature_df["rolling_std_v"] = (
        feature_df["rms_v"]
        .rolling(20, min_periods=1)
        .std()
        .fillna(0)
    )

    feature_df["health_index_h"] = (
        feature_df["rms_h"]
        /
        feature_df["rms_h"].iloc[0]
    )

    feature_df["health_index_v"] = (
        feature_df["rms_v"]
        /
        feature_df["rms_v"].iloc[0]
    )

    feature_df["combined_health"] = (
        feature_df["health_index_h"]
        +
        feature_df["health_index_v"]
    ) / 2

    feature_df["health_trend"] = (
        feature_df["combined_health"]
        .rolling(20, min_periods=1)
        .mean()
    )

    feature_df["damage_index"] = (
        feature_df["combined_health"]
    ).cumsum()

    feature_df[features] = scaler.transform(
        feature_df[features]
    )

    X_new = []

    for i in range(
        len(feature_df) - WIN_LEN + 1
    ):

        X_new.append(
            feature_df[features]
            .values[i:i+WIN_LEN]
        )

    X_new = np.array(X_new)

    X_new_tensor = torch.tensor(
        X_new,
        dtype=torch.float32
    )

    with torch.no_grad():

        pred = model(
            X_new_tensor
        )

    pred = (
        pred.numpy()
        .flatten()
    )

    predicted_rul = pred[-1]

    current_cycles = len(acc_files)

    estimated_cycles = int(
        predicted_rul * current_cycles
    )

    if predicted_rul >= 0.7:

        health = "Healthy"

        recommendation = (
            "The bearing is operating under healthy conditions. "
            "Continue regular monitoring."
        )

    elif predicted_rul >= 0.4:
 
        health = "Warning"

        recommendation = (
            "The bearing has entered the early degradation stage. "
            "Schedule an inspection."
        )

    elif predicted_rul >= 0.2:

        health = "Critical"

        recommendation = (
            "The bearing condition is critical. "
            "Plan bearing replacement soon."
        )

    else:

        health = "Replace Immediately"

        recommendation = (
            "The bearing is close to failure. "
            "Immediate replacement is recommended."
        )

    return (

        predicted_rul,

        estimated_cycles,

        health,

        recommendation,

        pred,

        feature_df
    )
