import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import os
from models.lstm_model import AttentivenessLSTM
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from config import Config

class DataGenerator:
    def __init__(self, video_dir, labels_file=None):
        self.config = Config()
        self.video_dir = video_dir
        self.labels_file = labels_file
        
        # Initialize detectors
        self.face_detector = FaceDetector(self.config.FACIAL_LANDMARKS_PATH)
        self.drowsiness_detector = DrowsinessDetector(
            self.config.EAR_THRESHOLD,
            self.config.MAR_THRESHOLD
        )
        self.distraction_detector = DistractionDetector(
            self.config.HEAD_POSE_THRESHOLD
        )
    
    def extract_features_from_video(self, video_path):
        features_sequence = []
        cap = cv2.VideoCapture(video_path)
        
        while len(features_sequence) < self.config.SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect face and landmarks
            face, landmarks = self.face_detector.detect(frame)
            if face is None:
                continue
            
            # Extract features
            features = self._extract_features(landmarks)
            features_sequence.append(features)
        
        cap.release()
        
        # Pad sequence if necessary
        while len(features_sequence) < self.config.SEQUENCE_LENGTH:
            features_sequence.append(np.zeros(6))  # 6 features per frame
            
        return np.array(features_sequence)
    
    def _extract_features(self, landmarks):
        # Extract drowsiness features
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        mouth = landmarks[48:60]
        
        ear_left = self.drowsiness_detector.calculate_ear(left_eye)
        ear_right = self.drowsiness_detector.calculate_ear(right_eye)
        mar = self.drowsiness_detector.calculate_mar(mouth)
        
        # Extract head pose features
        euler_angles = self.distraction_detector.get_head_pose(landmarks)
        
        # Combine features
        features = np.array([
            ear_left, ear_right, mar,
            euler_angles[0], euler_angles[1], euler_angles[2]
        ])
        
        return features
    
    def generate_dataset(self):
        X = []
        y = []
        
        # Load labels if available
        if self.labels_file and os.path.exists(self.labels_file):
            labels_df = pd.read_csv(self.labels_file)
        else:
            print("No labels file found. Creating dummy labels for demonstration.")
            video_files = os.listdir(self.video_dir)
            labels_df = pd.DataFrame({
                'video_file': video_files,
                'attentive': np.random.binomial(1, 0.7, len(video_files))  # 70% attentive
            })
        
        for _, row in labels_df.iterrows():
            video_path = os.path.join(self.video_dir, row['video_file'])
            if not os.path.exists(video_path):
                continue
                
            features = self.extract_features_from_video(video_path)
            X.append(features)
            y.append(row['attentive'])
        
        return np.array(X), np.array(y)

def train_model():
    config = Config()
    
    # Initialize data generator
    data_gen = DataGenerator(
        video_dir=os.path.join(config.VIDEO_DIR, 'sample_videos'),
        labels_file=os.path.join(config.VIDEO_DIR, 'labels.csv')
    )
    
    # Generate dataset
    print("Generating dataset...")
    X, y = data_gen.generate_dataset()
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    print("Training model...")
    model = AttentivenessLSTM(
        sequence_length=config.SEQUENCE_LENGTH,
        n_features=6  # Number of features per frame
    )
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS
    )
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, 'lstm_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history

if __name__ == "__main__":
    history = train_model()