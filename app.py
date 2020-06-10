import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

st.title("Dog Cat Classification")

try:
	uploaded_file = st.file_uploader("Choose an image of dog or cat...", type=["jpg","jpeg","png"])

	if uploaded_file is not None:
	 	
	    TARGET_SIZE = (224, 224)			
	    uploaded = Image.open(uploaded_file)	
	    model = load_model('DogCatClassifier.h5')
	    st.image(uploaded, caption = 'image uploaded...', use_column_width = True)
	    if st.button('Predict CLass'):
	    	img = np.array(uploaded.resize(TARGET_SIZE))/255
	    	img = np.expand_dims(img, axis = 0)
	    	result = int((model.predict(img)[0][0]) > 0.5)
	    	if result:
	    		st.write('This is a dog')
	    	else:
	    		st.write('This is a cat')
except:
	st.write("Corrupted Image!!...please try another")
