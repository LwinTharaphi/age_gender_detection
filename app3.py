# =========================
# PERFORMANCE & STABILITY
# =========================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# =========================
# IMPORTS
# =========================
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import MeanAbsoluteError
from PIL import Image, ImageDraw
import numpy as np
import cv2
import io
import base64

# =========================
# APP
# =========================
app = Flask(__name__)

# =========================
# LOAD MODELS (NO COMPILE)
# =========================
model_age = load_model(
    "age-gender.h5",
    custom_objects={"mae": MeanAbsoluteError()},
    compile=False
)

model_gender = load_model(
    "improved_age_gender_model_with_augmentation.h5",
    custom_objects={"mae": MeanAbsoluteError()},
    compile=False
)

# =========================
# LIGHTWEIGHT FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# HELPERS
# =========================
def detect_faces(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    faces_cv = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    faces = []
    for (x, y, w, h) in faces_cv:
        face = image.crop((x, y, x + w, y + h))
        faces.append((face, (x, y, x + w, y + h)))

    return faces


def predict_age_gender(face_img):
    img = face_img.convert("L").resize((128, 128))
    img = img_to_array(img).astype("float32") / 255.0
    img = img.reshape(1, 128, 128, 1)

    age_pred = model_age.predict(img, verbose=0)[1][0][0]
    gender_pred = model_gender.predict(img, verbose=0)[0][0]

    gender = "Female" if gender_pred > 0.5 else "Male"
    age = round(float(age_pred))

    return gender, age

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # LIMIT IMAGE SIZE (CRITICAL)
        MAX_SIZE = 800
        if max(img.size) > MAX_SIZE:
            img.thumbnail((MAX_SIZE, MAX_SIZE))

        faces = detect_faces(img)
        if not faces:
            return jsonify({"error": "No faces detected"})

        draw = ImageDraw.Draw(img)
        predictions = []

        for face, (x1, y1, x2, y2) in faces:
            gender, age = predict_age_gender(face)

            predictions.append({
                "gender": gender,
                "age": age,
                "box": [x1, y1, x2, y2]
            })

            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{gender}, {age}", fill="red")

        output = io.BytesIO()
        img.save(output, format="PNG")
        encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")

        return jsonify({
            "predictions": predictions,
            "image": encoded_image
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(debug=False)
