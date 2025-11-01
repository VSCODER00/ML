import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the model once (outside the Streamlit event)
model = load_model('CatsAndDogClassification.h5')

st.title("🐶🐱 Cats vs Dogs Classifier")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.resize((256, 256))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    st.write(prediction)
    # Display result
    if prediction > 0.5:
        st.success("🐶 It's a **Dog!**")
    else:
        st.success("🐱 It's a **Cat!**")
