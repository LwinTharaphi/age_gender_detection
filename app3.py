from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import MeanAbsoluteError
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import io

# -------------------- Flask App --------------------
app = Flask(__name__)

# -------------------- TensorFlow Optimization --------------------
tf.config.run_functions_eagerly(False)

# -------------------- Load Models --------------------
model_age = load_model(
    'age-gender.h5',
    custom_objects={'mae': MeanAbsoluteError()}
)

model_gender = load_model(
    'improved_age_gender_model_with_augmentation.h5',
    custom_objects={'mae': MeanAbsoluteError()}
)

# -------------------- Face Detector --------------------
detector = MTCNN()

# -------------------- Helpers --------------------
def resize_for_detection(image, max_size=800):
    w, h = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1:
        image = image.resize((int(w * scale), int(h * scale)))
    return image, scale


def detect_faces(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_np = np.array(image)
    results = detector.detect_faces(img_np)

    faces = []
    for r in results:
        if r["confidence"] < 0.95:
            continue

        x, y, w, h = r["box"]
        margin = int(max(w, h) * 0.1)

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_np.shape[1], x + w + margin)
        y2 = min(img_np.shape[0], y + h + margin)

        face = image.crop((x1, y1, x2, y2))
        faces.append((face, (x1, y1, x2, y2)))

    return faces[:3]  # limit max faces


def preprocess_face(face):
    face = face.convert("L")
    face = face.resize((128, 128))
    face = img_to_array(face)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


@tf.function
def predict_age(img):
    return model_age(img)


@tf.function
def predict_gender(img):
    return model_gender(img)


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    try:
        image = Image.open(io.BytesIO(file.read()))
        image, scale = resize_for_detection(image)

        faces = detect_faces(image)
        if not faces:
            return jsonify({"error": "No faces detected"})

        face_tensors = []
        boxes = []

        for face, box in faces:
            face_tensors.append(preprocess_face(face))
            boxes.append(box)

        face_tensors = np.vstack(face_tensors)

        # Batch prediction
        age_preds = predict_age(face_tensors)
        gender_preds = predict_gender(face_tensors)

        predictions = []
        for i in range(len(boxes)):
            age = int(round(float(age_preds[1][i][0])))
            gender = "Female" if gender_preds[0][i][0] > 0.5 else "Male"

            predictions.append({
                "gender": gender,
                "age": age,
                "box": boxes[i]
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------- Run Server --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True)
