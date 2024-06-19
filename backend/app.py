from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__, template_folder='../frontend/templates')
app.config['UPLOAD_FOLDER'] = 'backend/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_model.h5'
weights_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_model_weights.h5'
model = None

try:
    print(f"Attempting to load model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model architecture loaded successfully.")

    print(f"Attempting to load weights from {weights_path}")
    model.load_weights(weights_path)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading the model or weights: {e}")

# Helper function to preprocess video
def preprocess_video(video_path, frames_per_video=30, frame_size=(64, 64)):
    print(f"Preprocessing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading video.")
            break
        frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) == frames_per_video:
        print(f"Successfully preprocessed {frames_per_video} frames.")
        return np.array(frames)
    else:
        print(f"Only {len(frames)} frames were preprocessed, expected {frames_per_video}.")
        return None

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        print("Model is not loaded properly")
        return "Model is not loaded properly", 500

    if 'file' not in request.files:
        print("No file part in the request")
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to {file_path}")
        file.save(file_path)

        video = preprocess_video(file_path)
        if video is not None:
            print(f"Making prediction on the preprocessed video from {file_path}")
            prediction = model.predict(np.expand_dims(video, axis=0))[0, 0]
            label = 'Deepfake' if prediction > 0.5 else 'Real'
            print(f"Prediction: {prediction}, Label: {label}")
            return render_template('result.html', label=label, prediction=prediction, filename=file.filename)
        else:
            print("Failed to preprocess video")
    else:
        print("No file found")
    
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(f"Sending file {filename} from uploads folder")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("Starting Flask app")
    app.run(debug=True)
