import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🥔",
    layout="centered"
)

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .title {color: dir#2e7d32; text-align: center;}
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🥔 Potato Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload a potato leaf image to detect disease</p>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_my_model():
    return load_model("potato_disease_model.keras")

model = load_my_model()
classes = ["Early Blight", "Late Blight", "Healthy"]

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    st.divider()
    
    if st.button("🔍 Predict!", use_container_width=True):
        with st.spinner("Analyzing..."):
            img = Image.open(uploaded_file).resize((256, 256))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            result = classes[np.argmax(prediction)]
            confidence = round(float(np.max(prediction)) * 100, 2)
            
            if result == "Healthy":
                st.markdown(f"""
                    <div class='result-box' style='background-color:#e8f5e9; color:#2e7d32;'>
                        ✅ {result}<br>
                        <small>Confidence: {confidence}%</small>
                    </div>
                """, unsafe_allow_html=True)
                st.success("Potato plant is completely healthy!")
                
            elif result == "Early Blight":
                st.markdown(f"""
                    <div class='result-box' style='background-color:#fff3e0; color:#e65100;'>
                        ⚠️ {result}<br>
                        <small>Confidence: {confidence}%</small>
                    </div>
                """, unsafe_allow_html=True)
                st.warning("Early Blight detected! Spray fungicide as soon as possible!")
                
            else:
                st.markdown(f"""
                    <div class='result-box' style='background-color:#ffebee; color:#c62828;'>
                        🚨 {result}<br>
                        <small>Confidence: {confidence}%</small>
                    </div>
                """, unsafe_allow_html=True)
                st.error("Late Blight detected! Consult an agricultural expert immediately!")