import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Dog Cat Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
		
    uploaded = Image.open(uploaded_file)	
    img = image.load_img(uploaded_file, target_size = (224, 224))
    model = load_model('DogCatClassifier.h5')
    st.image(uploaded, caption = 'image uploaded...', use_column_width = True)
    img = np.array(img)/255
    img = np.expand_dims(img, axis = 0)
    result = int((model.predict(img)[0][0]) > 0.5)
    if result:
    	st.write('This is a dog')
    else:
    	st.write('This is a cat')
