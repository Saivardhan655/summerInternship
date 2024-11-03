import cv2
import numpy as np
import os
import pandas as pd
import logging
from tensorflow.keras.models import load_model
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from video_preprocessor import VideoPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)

class LSTMVideoTester:
    def __init__(self, model_path, feature_extractor):
        # Check if the model file exists
        if not os.path.isfile(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"File not found: {model_path}")
        
        # Load pre-trained LSTM model
        logging.info("Loading LSTM model...")
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
            
            # Verify all necessary features are present
            if features and all(k in features for k in ['avg_ear', 'avg_mar', 'avg_pitch', 'avg_roll', 'avg_yaw', 'dominant_emotion', 'drowsiness_ratio']):
                segment_features.append([
                    features['avg_ear'],
                    features['avg_mar'],
                    features['avg_pitch'],
                    features['avg_roll'],
                    features['avg_yaw'],
                    self.emotion_to_numeric(features['dominant_emotion']),
                    features['drowsiness_ratio']
                ])
            else:
                logging.warning(f"Missing features for segment {i}. Skipping this segment.")

        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

        # Step 2: Check if we have valid segment features
        if not segment_features:
            logging.error("No valid features extracted from the video. Prediction cannot proceed.")
            return None

        # Prepare data in the format LSTM expects
        segment_features = np.array(segment_features)
        
        # Check that the features match the expected shape for the model
        if segment_features.shape[1] != 9:
            logging.error(f"Expected 9 features per segment, but got {segment_features.shape[1]}")
            return None
        
        segment_features = np.expand_dims(segment_features, axis=0)  # Shape it to (1, num_segments, 9)

        # Step 3: Predict with LSTM model
        predictions = self.model.predict(segment_features)
        return predictions

    def emotion_to_numeric(self, emotion_label):
        # Define a mapping from emotion labels to numeric values if needed
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        return emotions.index(emotion_label) if emotion_label in emotions else -1

if __name__ == "__main__":
    # Define paths and instantiate classes
    lstm_model_path = "C:/Users/DELL/student attentivenes/data/trained_models/best_model.keras"
    video_path = "C:/Users/DELL/student attentivenes/data/videos/sample_videos/segment_0007.mp4"

    # Initialize the feature extractor and the tester
    feature_extractor = VideoPreprocessor()
    
    try:
        video_tester = LSTMVideoTester(lstm_model_path, feature_extractor)
        # Run predictions
        predictions = video_tester.process_and_predict(video_path)
        if predictions is not None:
            print("Predictions:", predictions)
        else:
            logging.error("Failed to make predictions due to missing or invalid input features.")
    except FileNotFoundError as e:
        logging.error(e)
