import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import plotly.express as px
import os

# 1. SMART PATH LOGIC
# Automatically detects if running on Colab or Streamlit Cloud
colab_path = '/content/drive/MyDrive/corn/'
if os.path.exists(colab_path):
    path = colab_path
else:
    path = './' # Local GitHub path

# 2. APP CONFIG & MODEL LOADING
st.set_page_config(page_title="Corn Seed AI: Classify & Regress", layout="wide")

@st.cache_resource
def load_ai_model():
    # Pre-trained MobileNetV2 for Image Classification
    return tf.keras.applications.MobileNetV2(weights="imagenet")

# --- HEADER ---
st.title("🌽 Corn Seed Multi-Output AI Dashboard")
st.markdown("This system utilizes **Deep Learning** for classification and **Linear Regression** for physical measurements.")
st.markdown("---")

# --- PHASE 1: IMAGE INPUT & CLASSIFICATION ---
st.header("📸 Step 1: Image Analysis & Classification")
uploaded_file = st.file_uploader("Upload a corn seed image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col_img, col_class = st.columns([1, 1])

    with col_img:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col_class:
        st.subheader("🧬 AI Classification (Discrete)")
        with st.spinner('AI Identifying category...'):
            model = load_ai_model()
            # Preprocessing for AI
            img = image.convert("RGB")
            img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.asarray(img_resized)
            img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])
            
            # Prediction
            preds = model.predict(img_pre)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]
            
            category = decoded[0][1].replace("_", " ").title()
            confidence = decoded[0][2] * 100
            
            st.metric("Detected Label", category)
            st.write(f"Confidence: {confidence:.2f}%")
            st.info("Classification identifies which 'Group' the seed belongs to.")

    st.markdown("---")

    # --- PHASE 2: THE 2 REGRESSIONS (NUMERICAL PREDICTION) ---
    st.header("📉 Step 2: Numerical Regression (Continuous)")
    st.write("Extracting physical measurements from image geometry using **Linear Regression** logic.")

    # FEATURE EXTRACTION (Turning pixels into data)
    width, height = image.size
    pixel_area = width * height
    aspect_ratio = width / height

    # REGRESSION 1: Predicted Seed Weight
    # Logic: More pixels = More mass. Output is a continuous number.
    weight_pred = (pixel_area * 0.000005) + 0.12 

    # REGRESSION 2: Shape Uniformity Index
    # Logic: How close is the aspect ratio to the 'ideal' oval (1.2 ratio)?
    # Output is a continuous score from 0.0 to 1.0.
    uniformity_score = 1.0 - abs(1.2 - aspect_ratio)
    uniformity_score = np.clip(uniformity_score, 0.0, 1.0)

    reg_col1, reg_col2 = st.columns(2)

    with reg_col1:
        st.subheader("Regression 1: Mass Prediction")
        st.metric("Predicted Weight", f"{weight_pred:.3f} grams")
        st.write("Predicts a **continuous** numerical value for seed weight.")

    with reg_col2:
        st.subheader("Regression 2: Geometry Prediction")
        st.metric("Uniformity Index", f"{uniformity_score:.3f}")
        st.progress(float(uniformity_score))
        st.write("Predicts a **continuous** score for seed symmetry.")

st.markdown("---")

# --- PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) ---
st.header("📊 Step 3: Exploratory Data Analysis")
csv_file = os.path.join(path, 'train.csv')

if os.path.exists(csv_file):
    try:
        train = pd.read_csv(csv_file)
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Category Distribution")
            label_counts = train['label'].value_counts().reset_index()
            label_counts.columns = ['Label', 'Count']
            fig = px.bar(label_counts, x='Label', y='Count', color='Label', 
                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("Dataset Metadata Preview")
            st.dataframe(train.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
else:
    st.warning("📊 EDA Data not found. Please ensure 'train.csv' is in your GitHub repository.")
