import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import time

st.title("Flower Classification")

try:
	uploaded_file = st.file_uploader("Choose an image of Daisy, Dandellion, Rose,Sunflower, or Tulip.....", 			type=["jpg","jpeg","png"])

	classes = ["Daisy","Dandelion","Rose","Sunflower","Tulip"]
	
	if uploaded_file is not None:
	    TARGET_SIZE = (224, 224)			
	    uploaded = Image.open(uploaded_file)
	    model = load_model('flower.h5')
	    st.image(uploaded, caption = 'image uploaded...', use_column_width = True)
	    actual_class = st.radio("What is the actual class?", classes)
	    if st.button('Predict Class'):
	    	img = np.array(uploaded.resize(TARGET_SIZE))/255
	    	img = np.expand_dims(img, axis = 0)
	    	result = classes[model.predict(img)[0].argmax()]
	    	
	    	st.write("This is a "+result)
	    	if result == actual_class:
	    	    st.write("Hurrah!!...Right Prediction")
	    	else:
	    	    st.write("Oh! NO...Wrong Prediction")
except:
	st.write("Corrupted image!!!...Plz try another")
