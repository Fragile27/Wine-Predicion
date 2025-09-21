import streamlit as st
import pandas as pd
import joblib

# ------------------- LOAD MODEL + SCALER -------------------
model = joblib.load("artifacts/wine_quality_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ------------------- STREAMLIT SETTINGS -------------------
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ------------------- MAIN UI -------------------
st.title("🍷 Wine Quality Prediction Dashboard")
st.markdown("<h3 style='color:#8B0000;'>A refined tool for predicting premium wine quality</h3>", unsafe_allow_html=True)

st.markdown("""
Welcome to the **Wine Quality Prediction App**!  
This tool uses a **Random Forest Classifier** trained on cleaned and balanced wine data.  
Use the sidebar to set wine chemistry attributes and discover if your wine is of premium quality.  
""")

# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    /* Predict button */
    .stButton>button {
        width: 60%;              /* increase width */
        height: 4em;             /* increase height */
        display: block;
        margin: 30px auto;
        font-size: 22px;         /* larger font */
        font-weight: 700;
        border-radius: 25px;     /* rounder corners */
        background: linear-gradient(90deg, #fd5949, #d6249f);
        color: white !important;
        border: none;
        box-shadow: 0 6px 25px rgba(0,0,0,0.35);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.08);
        box-shadow: 0 10px 35px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)


# ------------------- Sliders Layout -------------------
st.markdown("### ⚙️ Input Wine Measurements")
st.markdown("<p style='text-align:center;color:white;'>Adjust the sliders below to set wine attributes</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.0, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar", 0.0, 15.0, 1.9)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.076)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 80.0, 11.0)

with col2:
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
    density = st.slider("Density", 0.990, 1.005, 0.9978)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 9.4)

# ------------------- Predict Button & Result -------------------
if st.button("🍇 Predict Quality"):
    features = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.markdown(
            f"<div class='result-card good'>✅ Excellent! This wine is predicted to be <br><span style='font-size:28px;'>Good Quality 🍷</span><br>Confidence: {proba*100:.2f}%</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card bad'>❌ Unfortunately, this wine is predicted to be <br><span style='font-size:28px;'>Not Good Quality</span><br>Confidence: {(1-proba)*100:.2f}%</div>",
            unsafe_allow_html=True
        )

    st.markdown("### 📌 Your Entered Measurements")
    df = pd.DataFrame(features, columns=[
        "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
        "Chlorides", "Free SO₂", "Total SO₂", "Density", "pH", "Sulphates", "Alcohol"
    ])
    st.dataframe(df, use_container_width=True)

