import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from video_preprocessor import VideoPreprocessor

class LSTMVideoTester:
    def __init__(self, model_path, feature_extractor):
        # Load pre-trained LSTM model
        self.model = load_model(model_path)
        self.feature_extractor = feature_extractor

    def process_and_predict(self, video_path):
    # Directory to save video segments (temp storage for feature extraction)
        temp_dir = "temp_segments"
        os.makedirs(temp_dir, exist_ok=True)

        # Step 1: Split video and extract features for each segment
        num_segments = self.feature_extractor.split_video(video_path, temp_dir)
        segment_features = []

        for i in range(num_segments):
            segment_path = os.path.join(temp_dir, f'segment_{i:04d}.mp4')
            features = self.feature_extractor.extract_features(segment_path)
            # Adjust to match the 7 features your LSTM model expects
            segment_features.append([
                features['avg_ear'],
                features['avg_mar'],
                features['avg_pitch'],
                features['avg_roll'],
                features['avg_yaw'],
                self.emotion_to_numeric(features['dominant_emotion']),
                features['drowsiness_ratio']
            ])

        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        # Step 2: Prepare data in the format LSTM expects
        segment_features = np.array(segment_features)
        segment_features = np.expand_dims(segment_features, axis=0)  # Shape it to (1, num_segments, num_features)

        # Step 3: Predict with LSTM model
        predictions = self.model.predict(segment_features)
        return predictions


    def emotion_to_numeric(self, emotion_label):
        # Define a mapping from emotion labels to numeric values if needed
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        return emotions.index(emotion_label) if emotion_label in emotions else -1


if __name__ == "__main__":
    # Define paths and instantiate classes
    lstm_model_path ="C:/Users/DELL/student attentivenes/data/trained_models/lstm_model.h5"
    video_path = "C:/Users/DELL/student attentivenes/data/videos/sample_videos/attentive.mp4"

    # Initialize the feature extractor and the tester
    feature_extractor = VideoPreprocessor()
    video_tester = LSTMVideoTester(lstm_model_path, feature_extractor)

    # Run predictions
    predictions = video_tester.process_and_predict(video_path)
    print("Predictions:", predictions)
