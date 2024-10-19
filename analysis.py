import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load models
cnn_model = load_model('models/cnn_model.h5')
rf_model = joblib.load('models/random_forest_model.pkl')

# Load and preprocess a test image
test_image = cv2.imread('data/test/test_image.jpg')
test_image_resized = cv2.resize(test_image, (128, 128)) / 255.0
test_image_expanded = np.expand_dims(test_image_resized, axis=0)

# Predict with CNN
cnn_pred = np.argmax(cnn_model.predict(test_image_expanded), axis=1)
print("CNN Model Prediction:", cnn_pred)

# Predict with Random Forest
test_image_flat = test_image_resized.flatten().reshape(1, -1)
rf_pred = rf_model.predict(test_image_flat)
print("Random Forest Prediction:", rf_pred)
