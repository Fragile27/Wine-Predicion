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
# ------------------- CUSTOM STYLE -------------------
st.markdown("""
    <style>
    /* Background: Instagram gradient vibe */
    .stApp {
        background: linear-gradient(135deg, #fdf497, #fdf497 0%, #fd5949 50%, #d6249f 75%, #285AEB 100%);
        font-family: 'Arial', sans-serif;
        color: #111;
    }
    .block-container {
        max-width: 1000px;
        margin: auto;
        padding-top: 50px;
    }

    /* Title & description */
    h1, h3 {
        text-align: center;
        color: white;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.5);
    }
    p {
        text-align: center;
        color: white;
    }

    /* Sidebar hidden, sliders in main columns */
    section[data-testid="stSidebar"] {display:none;}

    /* Columns for sliders */
    .slider-col {
        display: flex;
        flex-direction: column;
        gap: 20px;
        width: 45%;
    }
    .sliders-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-bottom: 40px;
    }

    /* Sliders styling */
    .stSlider > div {
        background-color: rgba(255,255,255,0.9) !important;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stSlider label {
        font-weight: 600;
        color: #333;
    }

    /* Predict button */
    .stButton>button {
        width: 40%;
        display: block;
        margin: 20px auto;
        font-size: 18px;
        font-weight: 600;
        padding: 0.8em 1.5em;
        border-radius: 20px;
        background: linear-gradient(90deg, #fd5949, #d6249f);
        color: white !important;
        border: none;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    }

    /* Result cards */
    .result-card {
        margin-top: 30px;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .good {border-left: 6px solid #28a745;}
    .bad {border-left: 6px solid #dc3545;}

    /* Dataframe styling */
    .stDataFrame>div {
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        background-color: rgba(255,255,255,0.95);
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



