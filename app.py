import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
# from deployment import predict_image, label_encoder

from notebooks.deployment import predict_image , label_encoder


# Load model
model = load_model('./models/model_cnn.keras')


st.title("BrainScan AI - Tumor Classification")

# Upload image
uploaded_file = st.file_uploader("Select a brain MRI image", type=['jpg','jpeg','png','bmp'])

if uploaded_file is not None:
    # Convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display uploaded image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
    
    # Save temporarily and predict
    cv2.imwrite('temp.jpg', image)
    predicted_class = predict_image('temp.jpg', model, label_encoder=label_encoder)
    
    st.success(f"Predicted Class: {predicted_class}")
