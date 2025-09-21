import streamlit as st
import joblib
import numpy as np

# ================== Page Config ==================
st.set_page_config(
    page_title="💖 Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
    <style>
        /* Background */
        .stApp {
            background: linear-gradient(135deg, #ffe4ec, #ffc1e3, #ffb6c1);
            color: #4a0033;
            font-family: 'Poppins', sans-serif;
        }
        .block-container {
            max-width: 850px;
            margin: auto;
            padding-top: 40px;
        }

        /* Header */
        h1 {
            font-size: 3em;
            text-align: center;
            font-weight: 900;
            background: linear-gradient(90deg, #ff69b4, #ff1493, #ff6ec7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p {
            text-align: center;
            font-size: 1.2em;
            color: #660033;
            font-style: italic;
        }

        /* Dropdowns */
        .stSelectbox label {
            font-size: 1.1em !important;
            color: #550033 !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] {
            background-color: #ffe4ec !important;
            border-radius: 12px !important;
            border: 2px solid #ff69b4 !important;
            font-size: 1em !important;
            font-weight: 500 !important;
            color: #660033 !important;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff69b4, #ff1493, #ff6ec7);
            color: white !important;
            font-size: 20px !important;
            font-weight: 700 !important;
            padding: 1em 1.5em;
            border-radius: 30px;
            border: none;
            width: 100%;
            box-shadow: 0px 5px 25px rgba(255,105,180,0.7);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05) rotate(-1deg);
            box-shadow: 0px 10px 35px rgba(255,20,147,0.9);
        }

        /* Result card */
        .result-card {
            margin-top: 30px;
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 1.6em;
            font-weight: bold;
            font-family: 'Poppins', sans-serif;
        }
        .good {
            background: rgba(255,182,193,0.4);
            border: 3px solid #ff69b4;
            color: #b30059;
            box-shadow: 0px 0px 25px rgba(255,105,180,0.6);
        }
        .bad {
            background: rgba(255,228,225,0.6);
            border: 3px solid #ff1493;
            color: #800040;
            box-shadow: 0px 0px 25px rgba(255,20,147,0.6);
        }

        /* Extra Results */
        .extra-results {
            margin-top: 20px;
            padding: 15px;
            border-radius: 15px;
            background: rgba(255,240,245,0.8);
            font-size: 1.1em;
            color: #550033;
            box-shadow: 0px 0px 15px rgba(255,20,147,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>💖 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Pick your wine details, girl, and let’s see if it’s fabulous 🍷✨</p>", unsafe_allow_html=True)

# ================== Dropdown Inputs ==================
fixed_acidity = st.selectbox("Fixed Acidity", [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
volatile_acidity = st.selectbox("Volatile Acidity", [0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
citric_acid = st.selectbox("Citric Acid", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
residual_sugar = st.selectbox("Residual Sugar", [1, 2, 3, 5, 7, 10, 15])
chlorides = st.selectbox("Chlorides", [0.01, 0.05, 0.1, 0.15, 0.2])
free_sulfur_dioxide = st.selectbox("Free Sulfur Dioxide", [5, 10, 20, 30, 40, 50, 70])
total_sulfur_dioxide = st.selectbox("Total Sulfur Dioxide", [20, 50, 100, 150, 200, 250])
density = st.selectbox("Density", [0.990, 0.995, 1.000, 1.005])
pH = st.selectbox("pH", [2.8, 3.0, 3.2, 3.5, 3.8, 4.0])
sulphates = st.selectbox("Sulphates", [0.3, 0.5, 0.7, 1.0, 1.5])
alcohol = st.selectbox("Alcohol %", [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

# ================== Prediction ==================
if st.button("💅✨ Predict My Wine ✨💅"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Quality Result
    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>🌸✨ Premium Babe Wine! ✨🌸<br>Good Quality 💖🍷<br>Confidence: {probability[1]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>🙅‍♀️ Not Slaying Yet...<br>Needs a Glow-Up 💔🍷<br>Confidence: {probability[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    # Extra Breakdown Results
    st.markdown("""
        <div class='extra-results'>
            <b>🍬 Sweetness Level:</b> Based on Residual Sugar → Higher sugar = sweeter wine.<br>
            <b>🍋 Acidity Vibe:</b> Fixed & Volatile Acidity affect freshness.<br>
            <b>🍸 Alcohol Strength:</b> Higher % makes wine bolder.<br>
            <b>🌷 Overall Balance:</b> Perfect wines slay in all categories, babe! 💅✨
        </div>
    """, unsafe_allow_html=True)
