import cv2
import numpy as np
from config import Config
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector
from models.lstm_model import AttentivenessLSTM
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

class AttentivenessAnalyzer:
    def __init__(self):
        self.config = Config()
        
        # Initialize detectors
        self.face_detector = FaceDetector(self.config.FACIAL_LANDMARKS_PATH)
        self.drowsiness_detector = DrowsinessDetector(
            self.config.EAR_THRESHOLD,
            self.config.MAR_THRESHOLD
        )
        self.distraction_detector = DistractionDetector(
            self.config.HEAD_POSE_THRESHOLD
        )
        
        # Load LSTM model
        self.lstm_model = AttentivenessLSTM.load(
            os.path.join(self.config.MODEL_DIR, 'lstm_model.h5')
        )
        
        # Initialize buffers
        self.feature_buffer = deque(maxlen=self.config.SEQUENCE_LENGTH)
        self.prediction_buffer = deque(maxlen=self.config.MA_WINDOW)
        
    def process_frame(self, frame):
        # Detect face and landmarks
        face, landmarks = self.face_detector.detect(frame)
        if face is None:
            return frame, None
        
        # Extract features
        features = self._extract_features(landmarks)
        self.feature_buffer.append(features)
        
        # Make prediction if buffer is full
        prediction = None
        if len(self.feature_buffer) == self.config.SEQUENCE_LENGTH:
            X = np.array([list(self.feature_buffer)])
            prediction = self.lstm_model.predict(X)[0][0]
            self.prediction_buffer.append(prediction)
        
        # Calculate moving average
        if self.prediction_buffer:
            avg_prediction = np.mean(self.prediction_buffer)
        else:
            avg_prediction = None
        
        # Draw results on frame
        self._draw_results(frame, landmarks, avg_prediction)
        
        return frame, avg_prediction
    
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
    
    def _draw_results(self, frame, landmarks, prediction):
        # Draw facial landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw prediction
        if prediction is not None:
            status = "Focused" if prediction >= 0.5 else "Not Focused"
            cv2.putText(
                frame,
                f"Status: {status} ({prediction:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if prediction >= 0.5 else (0, 0, 255),
                2
            )
    
    def analyze_video(self, video_path=0):
        cap = cv2.VideoCapture(video_path)
        
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        predictions = []
        timestamps = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(
                frame,
                (self.config.DISPLAY_WIDTH, self.config.DISPLAY_HEIGHT)
            )
            
            # Process frame
            processed_frame, prediction = self.process_frame(frame)
            
            if prediction is not None:
                predictions.append(prediction)
                timestamps.append(len(predictions))
                
                # Update plots
                ax1.clear()
                ax2.clear()
                
                # Raw predictions
                ax1.plot(timestamps, predictions, 'b-')
                ax1.set_ylabel('Attentiveness Score')
                ax1.set_ylim([-0.1, 1.1])
                ax1.grid(True)
                
                # Moving average
                if len(predictions) >= self.config.MA_WINDOW:
                    ma = np.convolve(
                        predictions,
                        np.ones(self.config.MA_WINDOW)/self.config.MA_WINDOW,
                        mode='valid'
                    )
                    ma_timestamps = timestamps[self.config.MA_WINDOW-1:]
                    ax2.plot(ma_timestamps, ma, 'r-')
                
                ax2.set_xlabel('Frame')
                ax2.set_ylabel('Moving Average')
                ax2.set_ylim([-0.1, 1.1])
                ax2.grid(True)
                
                plt.pause(0.01)
            
            # Display frame
            cv2.imshow('Student Att')