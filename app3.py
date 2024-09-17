from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
import numpy as np
from PIL import Image, ImageDraw
import io
from tensorflow.keras.losses import MeanAbsoluteError
import cv2

app = Flask(__name__)

# Load both pre-trained models with custom objects
model_age = load_model('age-gender.h5', custom_objects={'mae': MeanAbsoluteError()})
model_gender = load_model('improved_age_gender_model_with_augmentation.h5', custom_objects={'mae': MeanAbsoluteError()})

# Initialize the face detector
detector = MTCNN()

def predict_age_gender(image):
    # Convert image to grayscale
    img = image.convert('L')
    img = img.resize((128, 128))  # Resize to match the model's expected input size
    img = img_to_array(img)
    
    # Add a channel dimension
    img = np.expand_dims(img, axis=-1)
    
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    
    img = img.astype('float32') / 255.0

    # Predict age using the age-focused model
    age_predictions = model_age.predict(img.reshape(1, 128, 128, 1))
    age_pred = age_predictions[1][0]  # Use the second output of the age model

    # Predict gender using the gender-focused model
    gender_predictions = model_gender.predict(img.reshape(1, 128, 128, 1))
    gender_pred = gender_predictions[0][0]  # Use the first output of the gender model

    # Decode the gender prediction
    gender = "Female" if gender_pred > 0.5 else "Male"
    
    # Ensure age_pred is a scalar
    age = round(float(age_pred[0]))  # Convert to float and then round

    return gender, age

def detect_faces(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image)
    results = detector.detect_faces(img_np)
    faces = []
    base_margin = 10

    for result in results:
        x, y, width, height = result['box']
        x2, y2 = x + width, y + height

        # Calculate margin relative to face size
        margin = int(max(width, height) * 0.1)  # 10% of face size

        x = max(0, x - margin)
        y = max(0, y - margin)
        x2 = min(img_np.shape[1], x2 + margin)
        y2 = min(img_np.shape[0], y2 + margin)

        face = image.crop((x, y, x2, y2))
        faces.append((face, (x, y, x2, y2)))

    return faces


@app.route('/')
def index():
    return render_template('index.html')

import base64

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        file.seek(0)
        img = Image.open(io.BytesIO(file.read()))

        # Detect faces
        faces = detect_faces(img)
        predictions = []
         

        if not faces:
            return jsonify({'error': 'No faces detected'})

        draw = ImageDraw.Draw(img)

        for face, (x, y, x2, y2) in faces:
            # Perform prediction
            gender, age = predict_age_gender(face)
            print(f"Predicted Gender: {gender}, Age: {age}")  # Debug: Print prediction results
            predictions.append({'gender': gender, 'age': age, 'box': [x, y, x2, y2]})

            # Draw rectangle around the face and text
            draw.rectangle([(x, y), (x2, y2)], outline="red", width=2)
            draw.text((x, y - 10), f"{gender}, {age}", fill="red")

        # Save the modified image in memory
        output = io.BytesIO()
        img.save(output, format='PNG')
        output.seek(0)
        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')

        # Return the image and predictions in response
        return jsonify({'predictions': predictions, 'image': encoded_image})

    except Exception as e:
        # More descriptive error message
        print(f"Error: {str(e)}")  # Debug: Print error message
        return jsonify({'error': f'Failed to process image: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=False)
