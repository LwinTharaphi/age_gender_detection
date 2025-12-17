Age and Gender Detection Using Deep Learning

This project is a web-based age and gender detection system that uses deep learning to predict a person's age and gender from images or real-time webcam input. The system is built with a Convolutional Neural Network (CNN) and deployed as an interactive web application.

🎯 Project Motivation

Age and gender detection has practical applications in areas such as:

Security and surveillance

Retail and customer analytics

Advertising and audience analysis

Human–computer interaction

This project demonstrates how deep learning can be applied to real-world computer vision problems.

🧠 Model Overview

Dataset: UTKFace Dataset

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Architecture: Multi-output model

Gender prediction (classification)

Age prediction (regression)

Outputs

Gender: Sigmoid activation (Male / Female)

Age: ReLU activation (predicted age value)

📊 Model Performance

Gender Accuracy: 83.54%

Age Prediction Error (MAE): 6.7 years

🖼️ Image Preprocessing & Prediction Workflow

Convert input images to grayscale

Detect and crop faces using OpenCV

Resize and normalize images

Feed preprocessed faces into the CNN model

Predict age and gender for each detected face

The system supports multiple face detection within a single image.

🌐 Web Application Features

Upload images for age and gender prediction

Real-time webcam integration

Display predicted age and gender results

Handle multiple faces in one image

🛠️ Tech Stack

Machine Learning

TensorFlow

Keras

OpenCV

Backend

Flask

Frontend

HTML

CSS

JavaScript

⚙️ Challenges & Solutions

Challenge: Handling multiple faces in a single image
Solution: Implemented OpenCV-based face detection to detect and predict each face individually

Challenge: Improving prediction accuracy
Solution: Applied image preprocessing techniques such as grayscale conversion and face cropping

🚀 How It Works (Webcam Mode)

Capture image from webcam

Detect faces in real time

Run predictions on detected faces

Display results instantly
