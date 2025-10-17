import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import cv2

# Load model and label classes
model = load_model('./models/model_cnn.keras')

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./models/classes.npy', allow_pickle=True)

def predict_image(image_path, model, image_size=(224,224), label_encoder=None):

    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    
    image = cv2.resize(image, image_size)

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]

    if label_encoder:
        class_label = label_encoder.inverse_transform([class_index])[0]
    else:
        class_label = class_index

    return class_label

image_path = './data/raw_data/glioma/Te-gl_0022.jpg'
predicted_class = predict_image(image_path, model, label_encoder=label_encoder)
print(f"Predicted class: {predicted_class}")

