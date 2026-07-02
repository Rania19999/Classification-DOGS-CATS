import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("Classification Chat vs Chien 🐱🐶")
st.write("Charge une image et le modèle prédit si c'est un chat ou un chien.")

@st.cache_resource
def load_my_model():
    return load_model('cats_dogs_model.keras')

model = load_my_model()

uploaded_file = st.file_uploader("Choisis une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Image chargée", use_column_width=True)

    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success(f"C'est un **chien** 🐶 (confiance : {prediction:.2%})")
    else:
        st.success(f"C'est un **chat** 🐱 (confiance : {1 - prediction:.2%})")