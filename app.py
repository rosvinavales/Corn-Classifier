import os

# 1. SMART PATH LOGIC
# This checks if the app is running on Colab (with Drive) or on the Web (GitHub)
colab_path = '/content/drive/MyDrive/corn/'
if os.path.exists(colab_path):
path = colab_path
@@ -20,14 +19,13 @@

@st.cache_resource
def load_ai_model():
    # Pre-trained MobileNetV2 (General Expert)
return tf.keras.applications.MobileNetV2(weights="imagenet")

# --- HEADER ---
st.title("🌽 Corn Seed Quality Analysis & AI Classifier")
st.markdown("---")

# --- SECTION 1: AI CLASSIFIER (THE PRIORITY) ---
# --- SECTION 1: AI CLASSIFIER ---
st.header("🤖 Phase 1: Real-Time AI Classification")
st.write("Upload an image of a corn seed to test the neural network's recognition.")

@@ -43,13 +41,11 @@ def load_ai_model():

with col2:
with st.spinner('Analyzing patterns...'):
            # AI Preprocessing: Fixes '4 vs 3 channel' error and resizes
img = image.convert("RGB")
img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
img_array = np.asarray(img)
img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_array[np.newaxis, ...])

            # AI Prediction
preds = model.predict(img_pre)
results = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

@@ -62,11 +58,10 @@ def load_ai_model():

st.markdown("---")

# --- SECTION 2: EXPLORATORY DATA ANALYSIS (THE RESEARCH) ---
# --- SECTION 2: EXPLORATORY DATA ANALYSIS (WITHOUT GALLERY) ---
st.header("📊 Phase 2: Exploratory Data Analysis (EDA)")
st.write("This section shows the statistical data used to understand seed quality categories.")

# Attempt to load the CSV data
csv_file = os.path.join(path, 'train.csv')

if os.path.exists(csv_file):
@@ -75,9 +70,9 @@ def load_ai_model():

# Summary Metrics
m1, m2, m3 = st.columns(3)
        m1.metric("Total Images in Dataset", len(train))
        m1.metric("Total Images in Dataset", f"{len(train):,}")
m2.metric("Quality Categories", len(train['label'].unique()))
        m3.metric("Deployment Environment", "Cloud" if path == './' else "Google Colab")
        m3.metric("Deployment Mode", "Cloud/GitHub" if path == './' else "Google Colab")

# Chart and Table
c1, c2 = st.columns([1, 1])
@@ -86,31 +81,16 @@ def load_ai_model():
label_counts = train['label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
fig = px.bar(label_counts, x='Label', y='Count', color='Label', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Distribution of Seed Types")
st.plotly_chart(fig, use_container_width=True)

with c2:
            st.subheader("Dataset Preview")
            st.subheader("Dataset Metadata (Sample)")
            st.write("This table shows how the images are labeled in our database:")
st.dataframe(train.head(10), use_container_width=True)

        # Gallery
        st.divider()
        st.subheader("Visual Sample Gallery")
        selected_cat = st.radio("Filter Gallery By Category:", train['label'].unique(), horizontal=True)
        
        # Show samples (only if images exist in the path)
        samples = train[train['label'] == selected_cat].head(6)
        gallery_cols = st.columns(6)
        
        for idx, (i, row) in enumerate(samples.iterrows()):
            img_p = os.path.join(path, row['image'])
            if os.path.exists(img_p):
                img_file = Image.open(img_p)
                gallery_cols[idx].image(img_file, caption=f"ID: {row['image']}", use_container_width=True)
            else:
                gallery_cols[idx].warning("Image file not found locally.")

except Exception as e:
st.error(f"Error loading CSV data: {e}")
else:
    st.warning(f"📊 EDA Data Not Found. To see the charts on the web, please upload 'train.csv' to your GitHub repository.")
    st.warning(f"📊 EDA Data Not Found. To see the charts, please upload 'train.csv' to your GitHub repository.")
