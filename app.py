import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ Force TensorFlow to use CPU only

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io  # ✅ Required to handle in-memory image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("hematovision_model.h5")

# Define class labels in the same order used during training
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def index():
    return render_template('index.html')  # Show image upload form

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # ✅ Fix: Read uploaded image using io.BytesIO
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = round(100 * np.max(predictions), 2)
    predicted_class = class_names[predicted_index]

    # Show result in result.html
    return render_template('result.html', result=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)