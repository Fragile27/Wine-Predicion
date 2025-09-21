import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title= "Wine Quality App", layout="wide")
# ================== Page Config ==================
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Load Model & Scaler ==================
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ================== Custom CSS ==================
st.markdown("""
   <style>
        /* 🎀 Clean Rounded Dropdown (No Highlights) */
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 25px !important;
            border: 2px solid #f78fb3 !important;
            background: #fff0f6 !important;
            font-size: 16px !important;
            box-shadow: 0px 3px 6px rgba(255, 120, 180, 0.2);
        /* Background */
        .stApp {
            background: linear-gradient(160deg, #0f0f0f, #1c1c1c);
            color: #f5f5f5;
            font-family: 'Segoe UI', sans-serif;
       }
        /* 🚫 Remove highlight completely */
        .stSelectbox div[data-baseweb="select"]:focus,
        .stSelectbox div[data-baseweb="select"]:hover {
            outline: none !important;
            background: #fff0f6 !important;
            border: 2px solid #f78fb3 !important;
            box-shadow: none !important;
        }
        /* 🔮 Dropdown menu itself */
        .stSelectbox ul {
            border-radius: 20px !important;
            background: #fff0f6 !important;
            border: 1px solid #f78fb3 !important;

        /* Center container */
        .block-container {
            max-width: 800px;
            margin: auto;
            padding-top: 40px;
       }
        .stSelectbox ul li {
            border-radius: 15px !important;

        /* Header */
        h1 {
            font-size: 3em;
            text-align: center;
            font-weight: 900;
            background: linear-gradient(90deg, #ff4b4b, #ff8888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
       }
        .stSelectbox ul li:hover {
            background: #ffd6eb !important;
            color: #d63384 !important;
        p {
            text-align: center;
            font-size: 1.2em;
            color: #cccccc;
       }
    </style>
""", unsafe_allow_html=True)

        /* Sliders */
        .stSlider > div > div {
            background: #333 !important;
            height: 10px;
            border-radius: 6px;
        }
        div[data-baseweb="slider"] span {
            background: #ff4b4b !important;
            border: 2px solid white !important;
            height: 24px !important;
            width: 24px !important;
            border-radius: 50%;
        }
        label {
            font-size: 1.1em !important;
            color: #eeeeee !important;
            font-weight: 600 !important;
        }

st.title("🍷Wine Quality App")

# Two-column layout (5:6 ratio)
col1, col2 = st.columns([5, 6])

with col1:
    fixed_acidity = st.selectbox("✨ Fixed Acidity", [round(x,1) for x in np.arange(4.0, 16.1, 0.5)])
    volatile_acidity = st.selectbox("💎 Volatile Acidity", [round(x,2) for x in np.arange(0.1, 1.6, 0.05)])
    citric_acid = st.selectbox("🌸 Citric Acid", [round(x,2) for x in np.arange(0.0, 1.1, 0.05)])
    residual_sugar = st.selectbox("🍬 Residual Sugar", [round(x,1) for x in np.arange(0.5, 15.1, 0.5)])
    chlorides = st.selectbox("🧂 Chlorides", [round(x,3) for x in np.arange(0.01, 0.21, 0.01)])
    free_sulfur_dioxide = st.selectbox("💨 Free Sulfur Dioxide", list(range(1, 73)))

with col2:
    total_sulfur_dioxide = st.selectbox("🌫️ Total Sulfur Dioxide", list(range(6, 290, 5)))
    density = st.selectbox("⚖️ Density", [round(x,3) for x in np.arange(0.990, 1.006, 0.001)])
    pH = st.selectbox("🧪 pH", [round(x,2) for x in np.arange(2.5, 4.6, 0.05)])
    sulphates = st.selectbox("🌟 Sulphates", [round(x,2) for x in np.arange(0.3, 2.1, 0.05)])
    alcohol = st.selectbox("🍹 Alcohol %", [round(x,1) for x in np.arange(8.0, 15.1, 0.5)])
        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #ff4b4b, #b22222);
            color: white !important;
            font-size: 20px !important;
            font-weight: 700 !important;
            padding: 0.8em 1.5em;
            border-radius: 10px;
            border: none;
            width: 100%;
            box-shadow: 0px 5px 20px rgba(255,75,75,0.5);
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 8px 30px rgba(255,75,75,0.8);
        }

# Predict Button
if st.button("💖✨ Predict Wine Quality ✨💖"):
    st.markdown('<div class="result-box">🍷 Your wine is likely to be of <span style="color:#ff4d79;">GOOD QUALITY</span> 💎✨</div>', unsafe_allow_html=True)
    st.markdown("🌸 Extra Results: This wine has balanced acidity, sweet notes, and perfect alcohol levels for a *premium taste* 💕🍇✨")
        /* Result card */
        .result-card {
            margin-top: 30px;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.6em;
            font-weight: bold;
        }
        .good {
            background: rgba(50,205,50,0.15);
            border: 2px solid #32cd32;
            color: #90ee90;
            box-shadow: 0px 0px 20px rgba(50,205,50,0.6);
        }
        .bad {
            background: rgba(255,75,75,0.15);
            border: 2px solid #ff4b4b;
            color: #ff9999;
            box-shadow: 0px 0px 20px rgba(255,75,75,0.6);
        }
    </style>
""", unsafe_allow_html=True)

# ================== App Header ==================
st.markdown("<h1>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Adjust the sliders below to test your wine blend.</p>", unsafe_allow_html=True)

# ================== Sliders ==================
fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.5)
chlorides = st.slider("Chlorides", 0.01, 0.2, 0.07)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 46)
density = st.slider("Density", 0.990, 1.005, 0.996, step=0.001)
pH = st.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
alcohol = st.slider("Alcohol %", 8.0, 15.0, 10.0)

# ================== Prediction ==================
if st.button("🔮 Predict Wine Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>✅ Premium Wine Detected!<br>Good Quality 🍷<br>Confidence: {probability[1]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Needs Refinement...<br>Not Good Quality 🍷<br>Confidence: {probability[0]*100:.2f}%</div>",
            unsafe_allow_html=True
        )
