import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from PIL import Image, ImageStat
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETUP
st.set_page_config(page_title="Corn AI: PNG Kernel Analysis", layout="wide")

# Path Configuration specifically for your folder structure
train_folder = './train'
csv_file = './train.csv'

# --- HELPER: EXTRACT KERNEL CHARACTERISTICS ---
def extract_kernel_features(img):
    # PNG Fix: Convert to RGB to remove alpha/transparency channel
    img = img.convert("RGB")
    width, height = img.size
    
    # Feature 1: Geometry
    area = width * height
    aspect_ratio = max(width, height) / min(width, height)
    
    # Feature 2: Color (using Red channel for seed health)
    stat = ImageStat.Stat(img)
    brightness = stat.mean[0] 
    
    # Feature 3: Texture (Crucial for identifying 'Broken' or 'Silkcut')
    # Standard deviation of pixels detects surface roughness
    texture_complexity = np.std(np.array(img))
    
    return [area, aspect_ratio, brightness, texture_complexity]

# --- THE MANDATORY ML PIPELINE (Professor's Checklist) ---
@st.cache_resource 
def build_model_pipeline():
    # STEP 1: Load dataset using Pandas
    if not os.path.exists(csv_file):
        st.error("Missing train.csv file!")
        return None, None, None, None, 0, 0, pd.DataFrame()
        
    df_meta = pd.read_csv(csv_file)
    
    # STEP 2: Handle missing values (Mode Imputation)
    df_meta['label'] = df_meta['label'].fillna(df_meta['label'].mode()[0])
    
    # STEP 3: Feature Selection & Extraction from PNGs
    real_data = []
    if not os.path.exists(train_folder):
        st.error(f"Folder '{train_folder}' not found!")
        return None, None, None, None, 0, 0, pd.DataFrame()

    available_images = [f for f in os.listdir(train_folder) if f.endswith('.png')]
    
    # Process up to 100 PNGs to build the model logic
    for img_name in available_images[:100]: 
        try:
            # Match filename with CSV label
            match = df_meta[df_meta['image'] == img_name]
            if not match.empty:
                label = match['label'].values[0]
                with Image.open(os.path.join(train_folder, img_name)) as img:
                    features = extract_kernel_features(img)
                    # Regression Target: Continuous Weight (Area * Density factor)
                    weight = (features[0] * 0.000005) + (features[2] * 0.001)
                    real_data.append(features + [weight, label])
        except:
            continue

    # Create Training DataFrame
    if len(real_data) < 10:
        st.error("Not enough PNG images found matching the CSV labels to train.")
        return None, None, None, None, 0, 0, pd.DataFrame()
        
    df = pd.DataFrame(real_data, columns=['Area', 'Aspect_Ratio', 'Brightness', 'Texture', 'Weight', 'Label'])
    
    # STEP 4: Encoding (Categorical to Numerical)
    le = LabelEncoder()
    df['Label_Encoded']
