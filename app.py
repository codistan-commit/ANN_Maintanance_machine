import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_artifacts()


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }

        .title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #777777;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        .result-box {
            padding: 1.4rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 1.5rem;
        }

        .normal {
            background-color: #e8f5e9;
            border: 1px solid #a5d6a7;
            color: #1b5e20;
        }

        .failure {
            background-color: #ffebee;
            border: 1px solid #ef9a9a;
            color: #b71c1c;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: inherit;
        }

        .result-text {
            font-size: 1rem;
            margin-top: 0.5rem;
            color: inherit;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">Machine Failure Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">ANN-based predictive maintenance system</div>',
    unsafe_allow_html=True
)


st.divider()


# -----------------------------
# Input Section
# -----------------------------
st.subheader("Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    vibration = st.number_input(
        "Vibration",
        min_value=0.0,
        value=0.8,
        step=0.01
    )

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        value=65.0,
        step=0.1
    )

    imf_1 = st.number_input(
        "IMF 1",
        value=0.16,
        step=0.001,
        format="%.4f"
    )

    imf_3 = st.number_input(
        "IMF 3",
        value=0.001,
        step=0.001,
        format="%.4f"
    )


with col2:
    acoustic = st.number_input(
        "Acoustic",
        min_value=0.0,
        value=0.60,
        step=0.01
    )

    current = st.number_input(
        "Current",
        min_value=0.0,
        value=12.0,
        step=0.1
    )

    imf_2 = st.number_input(
        "IMF 2",
        value=0.0,
        step=0.001,
        format="%.4f"
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Machine Status", use_container_width=True):

    # Keep EXACT same feature order used during training
    input_data = np.array([[
        vibration,
        acoustic,
        temperature,
        current,
        imf_1,
        imf_2,
        imf_3
    ]])

    # Convert to DataFrame with the same feature names
    input_df = pd.DataFrame(
        input_data,
        columns=[
            "vibration",
            "acoustic",
            "temperature",
            "current",
            "IMF_1",
            "IMF_2",
            "IMF_3"
        ]
    )

    # Apply the SAME scaler used during training
    input_scaled = scaler.transform(input_df)

    # Prediction
    probability = model.predict(input_scaled, verbose=0)[0][0]

    prediction = 1 if probability >= 0.5 else 0

    # -----------------------------
    # Display Result
    # -----------------------------
    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-box failure">
                <div class="result-title">⚠️ Machine Failure Detected</div>
                <div class="result-text">
                    Failure probability: <b>{probability * 100:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-box normal">
                <div class="result-title">✓ Machine Operating Normally</div>
                <div class="result-text">
                    Failure probability: <b>{probability * 100:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Predictive Maintenance • Artificial Neural Network"
)