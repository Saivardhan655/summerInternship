import os
import cv2
import dlib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load Dlib model for facial landmarks and pre-trained emotion model
print("Loading models...")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:/Users/DELL/summerInternship/models/shape_predictor_68_face_landmarks.dat")
    emotion_model = load_model("C:/Users/DELL/summerInternship/models/emotion_model.h5")
    print("Models loaded successfully.")
except Exception as e:
    print("Error loading models:", e)

# Define EAR and MAR calculations
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])
    B = np.linalg.norm(mouth_points[4] - mouth_points[8])
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])
    return (A + B) / (2.0 * C)

# Main feature extraction function
def extract_features(video_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        print(f"Detected faces: {faces}")

        for face in faces:
            landmarks = predictor(gray, face)
            print(f"Processing face: {face}")

            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            face_crop = gray[face.top():face.bottom(), face.left():face.right()]
            face_resized = cv2.resize(face_crop, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=[0, -1])
            emotion = emotion_model.predict(face_resized)[0]

            feature_vector = [ear, mar, *emotion]
            features.append(feature_vector)
            print(f"Features collected: {feature_vector}")

    cap.release()
    print(f"Feature extraction complete for video: {video_path}")
    return features

def save_features(video_files):
    print("Save feature called")
    all_features = []
    for video_file in video_files:
        features = extract_features(video_file)
        all_features.extend(features)

    print(f"Total number of features collected: {len(all_features)}")

    if len(all_features) == 0:
        print("No features collected; nothing to save.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(all_features, columns=['EAR', 'MAR', 'Emotion1', 'Emotion2', 'Emotion3', 'Emotion4', 'Emotion5', 'Emotion6', 'Emotion7'])
        print("DataFrame content before saving:\n", df.head())
        
        # Save to CSV
        try:
            output_file = "C:/Users/DELL/summerInternship/data/labeled_features.csv"
            df.to_csv(output_file, index=False)
            print(f"Features saved to CSV at {output_file}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

# List of videos to process
video_files = ["C:/Users/DELL/summerInternship/data/videos/video2.mp4"]
save_features(video_files)
