import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import plotly.express as px

# 1. SETUP & PATHS
st.set_page_config(page_title="Corn Seed AI Dashboard", layout="wide")

path = '/content/drive/MyDrive/corn/'

@st.cache_resource
def load_ai_model():
    # Pre-trained MobileNetV2
    return tf.keras.applications.MobileNetV2(weights="imagenet")

# --- HEADER ---
st.title("🌽 Corn Seed Quality Analysis & AI Classifier")
st.markdown("---")

# --- SECTION 1: AI CLASSIFIER (THE PRIORITY) ---
st.header("🤖 Phase 1: Real-Time AI Classification")
st.write("Upload an image of a corn seed to test the neural network's recognition.")

model = load_ai_model()
uploaded_file = st.file_uploader("Drop a seed image here", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Target Image", use_container_width=True)

    with col2:
        with st.spinner('Analyzing patterns...'):
            # AI Preprocessing
            img = image.convert("RGB")
            img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.asarray(img)
            img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])

            # AI Prediction
            preds = model.predict(img_pre)
            results = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

        st.success("AI Analysis Complete!")
        for i in results:
            label = i[1].replace("_", " ").title()
            confidence = i[2]*100
            st.write(f"**{label}**: {confidence:.2f}%")
            st.progress(int(confidence))

st.markdown("---")

# --- SECTION 2: EXPLORATORY DATA ANALYSIS (THE RESEARCH) ---
st.header("📊 Phase 2: Exploratory Data Analysis (EDA)")
st.write("This section shows the data used to understand seed quality categories.")

try:
    # Load the CSV
    train = pd.read_csv(path + 'train.csv')

    # Summary Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Images", len(train))
    m2.metric("Categories", len(train['label'].unique()))
    m3.metric("Data Source", "Kaggle POG")

    # Chart and Table
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Category Distribution")
        # Fancy Plotly Bar Chart
        label_counts = train['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']
        fig = px.bar(label_counts, x='Label', y='Count', color='Label',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Dataset Preview")
        st.dataframe(train.head(10), use_container_width=True)

    # Gallery
    st.divider()
    st.subheader("Visual Sample Gallery")
    selected_cat = st.radio("Filter Gallery By Category:", train['label'].unique(), horizontal=True)

    samples = train[train['label'] == selected_cat].head(6)
    gallery_cols = st.columns(6)

    for idx, (i, row) in enumerate(samples.iterrows()):
        img_p = path + row['image']
        img_file = Image.open(img_p)
        gallery_cols[idx].image(img_file, caption=f"ID: {row['image']}", use_container_width=True)

except Exception as e:
    st.error(f"Waiting for Data Connection... (Check your folder path: {path})")
