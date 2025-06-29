import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model("hematovision_model.h5")

# Define class labels (same order as your training dataset)
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Path to test images folder
test_folder = "test_images"
image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png'))]

# Predict each image
for img_name in image_files:
    img_path = os.path.join(test_folder, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = round(100 * np.max(predictions), 2)

    print(f"\n📷 {img_name}")
    print("🧠 Predicted Cell Type:", class_names[predicted_class])
    print("✅ Confidence:", confidence, "%")