import cv2
import dlib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
emotion_model = load_model("../models/emotion_model.h5")

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

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            face_crop = gray[face.top():face.bottom(), face.left():face.right()]
            face_resized = cv2.resize(face_crop, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=[0, -1])
            emotion = emotion_model.predict(face_resized)[0]

            features.append([ear, mar, *emotion])
    cap.release()
    return features

# Save extracted features to CSV for further use
def save_features(video_files):
    all_features = []
    for video_file in video_files:
        features = extract_features(video_file)
        all_features.extend(features)
    pd.DataFrame(all_features).to_csv("../data/labeled_features.csv", index=False)


if __name__ == "__main__":
    video_files = ["../data/videos/Sidharth vs Vinayak (Online classes gone wrong 🤣).mp4"]
    if(video_files):
       print('video file found')
    save_features(video_files)
