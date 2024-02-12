import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ======================== Setup, Styling and Layout Adjustments =========================

# Make page wide
st.set_page_config(layout="wide")

# Hide the menu button and Streamlit icon on the footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       /* Set a max width for the main content area and center it */
       .main .block-container {
           max-width: 800px;  /* You can adjust this value */
           padding: 2rem 1rem;
           margin: auto;
       }
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Change font to Catamaran
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Catamaran:wght@100;200;300;400;500;600;700;800;900&display=swap');
            html, body, [class*="css"]  {
            font-family: 'Catamaran', sans-serif;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

# ======================== Model and Functions =========================

# Load your trained model
best_model = load_model('EfficientNetB3_model_67.h5')
#best_model = load_model('EfficientNetB3_model_67_full.h5')

def preprocess_image(uploaded_file):
    # Load the uploaded image
    img = Image.open(uploaded_file)

    # Resize the image to match the model's expected input size
    img = img.resize((224, 224))

    # Convert the image to an array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for EfficientNetB3
    img_array = preprocess_input(img_array)

    return img_array

def predict_emotion(model, img_array):
    # Use the model to predict the emotion
    predictions = model.predict(img_array)

    # Decode the prediction
    emotion_index = np.argmax(predictions)
    emotions = ['Neutral', 'Happy', 'Angry', 'Sad', 'Neutral Dog', 'Happy Dog', 'Angry Dog', 'Sad Dog']
    return emotions[emotion_index]

# ======================== Streamlit App Layout =========================

# Header and Sub-header
st.markdown("<h1 style='text-align: center; color: black;'>🐾 Pet Emotion Recognition Tool 🐾</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Upload a picture of your cat or dog to detect its emotion</h3>", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Please wait for the emotion prediction below...', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(uploaded_file)

    # Predict emotion
    predicted_emotion = predict_emotion(best_model, img_array)

    # Display the prediction
    st.write("Predicted Emotion:", predicted_emotion)
