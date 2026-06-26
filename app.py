import io
import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib import colors

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.units import inch
import numpy as np
import pandas as pd
import streamlit as st
import torch
from model import CNN_LSTM
from feature_extraction import extract_vibration_features
from reportlab.lib.utils import ImageReader
FEATURE_COUNT = 21

model = CNN_LSTM(
    input_features=FEATURE_COUNT
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

model.eval()

st.set_page_config(
    page_title="Bearing RUL Prediction System",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Bearing Remaining Useful Life Prediction System")

st.markdown(
    """
Predict the Remaining Useful Life (RUL) of rolling bearings using
a CNN-LSTM deep learning model based on vibration signal analysis.
"""
)

st.success("✅ CNN-LSTM model loaded successfully.")

st.sidebar.title("⚙️ Model Information")

st.sidebar.write("**Model:** CNN-LSTM")

st.sidebar.write("**Features:** 21")

st.sidebar.write("**Window Size:** 20")

st.sidebar.write("**Framework:** PyTorch")

st.sidebar.write("**Dataset:** FEMTO Bearing Dataset")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload or select a bearing folder to predict its Remaining Useful Life."
)

import streamlit as st
import glob
import os

folder_path = st.text_input(
    "Enter Bearing Folder Path"
)

if st.button("Read Files"):

    acc_files = sorted(
        glob.glob(
            os.path.join(
                folder_path,
                "acc_*.csv"
            )
        )
    )

    st.write(
        f"Found {len(acc_files)} CSV files"
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

    import pickle

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

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    feature_df[features] = scaler.transform(
        feature_df[features]
    )

    win_len = 20

    X_new = []

    for i in range(
        len(feature_df) - win_len + 1
    ):

        X_new.append(
            feature_df[features]
            .values[i:i+win_len]
        )

    X_new = np.array(X_new)

    st.write(
        "Sequence Shape:",
        X_new.shape
    )
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


    st.markdown("---")

    st.subheader("🔧 Maintenance Recommendation")

    if predicted_rul >= 0.7:

        health = "🟢 Healthy"

        recommendation = (
            "The bearing is operating under healthy conditions. "
            "No immediate maintenance is required. "
            "Continue regular monitoring and preventive maintenance."
        )

        st.success(recommendation)

    elif predicted_rul >= 0.4:

        health = "🟡 Warning"

        recommendation = (
            "The bearing has entered the early degradation stage. "
            "Plan an inspection during the next maintenance cycle "
            "to avoid unexpected deterioration."
        )

        st.warning(recommendation)

    elif predicted_rul >= 0.2:

        health = "🟠 Critical"

        recommendation = (
            "The bearing condition is critical. "
            "Maintenance should be scheduled as soon as possible. "
            "Continued operation may increase the risk of failure."
        )

        st.warning(recommendation)

    else:

        health = "🔴 Replace Immediately"

        recommendation = (
            "The bearing is close to the end of its useful life. "
            "Immediate replacement is strongly recommended to prevent equipment failure."
        )

        st.error(recommendation)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📈 Predicted RUL")

        st.metric(
            label="Normalized RUL",
            value=f"{predicted_rul:.3f}"
        )

        st.progress(
            float(min(max(predicted_rul, 0), 1))
        )

    with col2:

        st.markdown("### 🔄 Remaining Cycles")

        st.metric(
            label="Estimated",
            value=f"{estimated_cycles}"
        )

    with col3:

        st.markdown("### ❤️ Health Status")

        if predicted_rul >= 0.7:

            st.success("Healthy")

        elif predicted_rul >= 0.4:

            st.warning("Warning")

        elif predicted_rul >= 0.2:

            st.warning("Critical")

        else:

            st.error("Replace Immediately")
    st.markdown("---")

    st.subheader("📂 Bearing Information")

    left, right = st.columns(2)

    with left:

        st.write("**Bearing Name**")
        st.info(os.path.basename(folder_path))

        st.write("**CSV Files Processed**")
        st.info(len(acc_files))

        st.write("**Window Size**")
        st.info(win_len)

    with right:

        st.write("**Model Used**")
        st.info("CNN-LSTM")

        st.write("**Features Used**")
        st.info(len(features))

        st.write("**Prediction Time**")
        st.info(
            datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )

    st.markdown("---")

    st.subheader("📈 Remaining Useful Life Trend")

    graph_df = pd.DataFrame({

        "Cycle": np.arange(
            win_len,
            len(acc_files)+1
        ),

        "Predicted RUL": pred
    })

    import plotly.express as px

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        graph_df["Cycle"],
        graph_df["Predicted RUL"],
        linewidth=2
    )

    ax.set_title("Remaining Useful Life Trend")

    ax.set_xlabel("Bearing Cycle")

    ax.set_ylabel("Normalized Remaining Useful Life")

    ax.grid(True)

    st.pyplot(fig)

    graph_buffer = io.BytesIO()

    fig.savefig(
        graph_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    graph_buffer.seek(0)

    styles = getSampleStyleSheet()

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(pdf_buffer)

    elements = []

    elements.append(
        Paragraph(
            "<b>Bearing Remaining Useful Life Prediction Report</b>",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"Date : {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Bearing : {os.path.basename(folder_path)}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            "Model : CNN-LSTM",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1,12))

    from reportlab.lib.styles import ParagraphStyle

    wrap_style = ParagraphStyle(
        "WrapStyle",
        parent=styles["BodyText"],
        leading=16
    )

    table_data = [

        ["Parameter","Value"],

        ["Predicted RUL",
         f"{predicted_rul:.4f}"],

        ["Remaining Cycles",
         str(estimated_cycles)],

        ["Health Status",
         health],

        ["Recommendation",
         Paragraph(
             recommendation,
             wrap_style
             )]
    ]

    table = Table(table_data,
                  colWidths=[140,320]
                  )

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.grey),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,1),(-1,-1),colors.beige),

            ("BOTTOMPADDING",(0,0),(-1,0),10),

            ("VALIGN", (0,0), (-1,-1), "TOP"),

        ])

    )

    elements.append(table)

    elements.append(Spacer(1,20))

    elements.append(
        Paragraph(
            "<b>Remaining Useful Life Trend</b>",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1,10))

    graph_image = Image(
        graph_buffer,
        width=6*inch,
        height=3*inch
    )

    elements.append(graph_image)

    elements.append(Spacer(1,20))

    elements.append(
        Paragraph(
            "<b>Generated by Bearing Remaining Useful Life Prediction System</b>",
            styles["Normal"]
        )
    )

    doc.build(elements)

    pdf_buffer.seek(0)

    st.download_button(

        label="📄 Download PDF Report",

        data=pdf_buffer,

        file_name="Bearing_RUL_Report.pdf",

        mime="application/pdf"
    )
