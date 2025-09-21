import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Background & Font */
        .stApp {
            background-color: #f5f5f5;
            color: #111;
            font-family: 'Arial', sans-serif;
        }
        .block-container {
            max-width: 700px;
            margin: auto;
            padding-top: 50px;
        }

        /* Header */
        h1 {
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            color: #111;
            margin-bottom: 10px;
        }
        p {
            text-align: center;
            font-size: 1.1em;
            color: #333;
            margin-bottom: 40px;
        }

        /* Sliders centered */
        .stSlider {
            width: 80% !important;
            margin: auto;
        }

        /* Button */
        .stButton>button {
            background-color: #111;
            color: white !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            padding: 0.8em 1.5em;
            border-radius: 10px;
            width: 50%;
            display: block;
            margin: 30px auto;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #333;
            transform: scale(1.03);
        }

        /* Result card */
        .result-card {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            background-color: #fff;
            color: #111;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
        .good {
            border-left: 5px solid #28a745;
        }
        .bad {
            border-left: 5px solid #dc3545;
        }

        /* Extra Results */
        .extra-results {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #fff;
            font-size: 1em;
            color: #111;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Adjust the sliders and see the predicted wine quality.</p>", unsafe_allow_html=True)

# ================== Slider Inputs ==================
fixed_acidity = st.slider("Fixed Acidity", 4.0, 10.0, 7.0, 0.1)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5, 0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
residual_sugar = st.slider("Residual Sugar", 0, 15, 5, 1)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05, 0.01)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 5, 70, 20, 1)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 20, 250, 100, 5)
density = st.slider("Density", 0.990, 1.005, 0.995, 0.001)
pH = st.slider("pH", 2.8, 4.0, 3.2, 0.01)
sulphates = st.slider("Sulphates", 0.3, 1.5, 0.6, 0.01)
alcohol = st.slider("Alcohol %", 8.0, 15.0, 11.0, 0.1)

# ================== Prediction ==================
if st.button("Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Quality Result
    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>Premium Wine 🍷<br>Confidence: {probability[1]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>Low Quality Wine 🍷<br>Confidence: {probability[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Extra Breakdown Results
    st.markdown(f"""
        <div class='extra-results'>
            <b>Sweetness:</b> Residual Sugar level.<br>
            <b>Acidity:</b> Fixed & Volatile Acidity affect freshness.<br>
            <b>Alcohol:</b> Higher % makes wine bolder.<br>
            <b>Balance:</b> Wines are best when all features align.
        </div>
    """, unsafe_allow_html=True)
