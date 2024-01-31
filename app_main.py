import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Setup and layout adjustments
st.set_page_config(layout="wide", page_title="Pet Emotion Recognition Tool")

# App layout
st.markdown("<h1 style='text-align: center; color: yellow;'>🐾 Pet Emotion Recognition Tool 🐾</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Upload a picture of your cat or dog to detect its emotion</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a clear picture of your pet's face.")

# Load model
best_model = load_model('EfficientNetB3_model_67.h5')

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_emotion(model, img_array):
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    emotions = ['Normal', 'Happy', 'Angry', 'Sad', 'Normal', 'Happy', 'Angry', 'Sad']
    return emotions[emotion_index]

if uploaded_file is not None:
    # Use columns to center the image on larger screens and allow it to fill the space on smaller ones
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Check Emotion", help="Click to detect your pet's emotion"):
        with st.spinner('Analyzing...'):
            img_array = preprocess_image(uploaded_file)
            predicted_emotion = predict_emotion(best_model, img_array)
            st.markdown(f"<h2 style='text-align: center;'>Your pet seems <span style='color: #4CAF50;'>{predicted_emotion}</span>!</h2>", unsafe_allow_html=True)
