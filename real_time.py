import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from collections import deque
from sklearn.preprocessing import StandardScaler
import os
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector

class RealtimeAttentionMonitor:
    def __init__(self, lstm_model_path, emotion_model_path):
        # Load LSTM model for attention prediction
        self.lstm_model = load_model(lstm_model_path, compile=False)
        
        # Initialize feature extractors
        self.face_detector = FaceDetector('data/trained_models/shape_predictor_68_face_landmarks.dat')
        self.drowsiness_detector = DrowsinessDetector(ear_threshold=0.3, mar_threshold=0.6)
        self.distraction_detector = DistractionDetector(pose_threshold=30)
        
        # Load emotion detection model
        self.emotion_model = load_model(emotion_model_path)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Buffer for features
        self.feature_buffer = deque(maxlen=5)  # Adjusted for a sequence length of 5
        self.frame_features = {
            'ear_values': [],
            'mar_values': [],
            'pitch': [],
            'roll': [],
            'yaw': [],
            'emotions': []
        }
        
        self.scaler = StandardScaler()
        
        # Metrics for display
        self.metrics = {
            'face_status': 'No Face Detected',
            'attentiveness': 'Unknown',
            'attention_score': 0.0,
            'current_emotion': 'Unknown',
            'drowsiness': 'Alert',
            'distraction': 'Focused'
        }

    def _extract_face_roi(self, frame, face):
        """Extract and preprocess face ROI for emotion detection"""
        x, y, w, h = face.left(), face.top(), face.height(), face.height()
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)
        return face_roi

    def _aggregate_features(self):
        """Aggregate features from the current frame"""
        if not self.frame_features['emotions']:
            return None
            
        # Calculate emotion statistics
        emotion_counts = pd.Series(self.frame_features['emotions']).value_counts()
        
        # Get average values
        features = {
            'avg_ear': np.mean(self.frame_features['ear_values']),
            'avg_mar': np.mean(self.frame_features['mar_values']),
            'avg_pitch': np.mean(self.frame_features['pitch']),
            'avg_roll': np.mean(self.frame_features['roll']),
            'avg_yaw': np.mean(self.frame_features['yaw']),
            'dominant_emotion': emotion_counts.index[0] if not emotion_counts.empty else 'Unknown',
            'emotion_diversity': len(emotion_counts),
            'drowsiness_ratio': np.mean(np.array(self.frame_features['ear_values']) < 0.3),
            'distraction_ratio': np.mean(np.abs(self.frame_features['yaw']) > 30)
        }
        
        # Clear frame features
        for key in self.frame_features:
            self.frame_features[key] = []
            
        return features

    def process_frame(self, frame):
        try:
            face, landmarks = self.face_detector.detect(frame)
            if face is not None:
                self.metrics['face_status'] = 'Face Detected'
                x, y, w, h = face.left(), face.top(), face.height(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Extract drowsiness features
                left_eye = landmarks[42:48]
                right_eye = landmarks[36:42]
                mouth = landmarks[48:60]

                ear_left = self.drowsiness_detector.calculate_ear(left_eye)
                ear_right = self.drowsiness_detector.calculate_ear(right_eye)
                ear_avg = (ear_left + ear_right) / 2
                mar = self.drowsiness_detector.calculate_mar(mouth)

                # Extract head pose features
                euler_angles = self.distraction_detector.get_head_pose(landmarks)

                # Extract emotion
                face_roi = self._extract_face_roi(frame, face)
                emotion_pred = self.emotion_model.predict(face_roi, verbose=0)
                emotion_label = self.emotions[np.argmax(emotion_pred)]

                # Store features
                self.frame_features['ear_values'].append(ear_avg)
                self.frame_features['mar_values'].append(mar)
                self.frame_features['pitch'].append(euler_angles[0])
                self.frame_features['roll'].append(euler_angles[1])
                self.frame_features['yaw'].append(euler_angles[2])
                self.frame_features['emotions'].append(emotion_label)

                # Aggregate features and make prediction
                features = self._aggregate_features()
                if features:
                    # Update metrics for current emotion
                    self.metrics['current_emotion'] = features['dominant_emotion']
                    
                    # Update drowsiness and distraction status based on ratios
                    drowsiness_ratio = features['drowsiness_ratio']
                    distraction_ratio = features['distraction_ratio']
                    self.metrics['drowsiness'] = 'Drowsy' if drowsiness_ratio > 0.5 else 'Alert'
                    self.metrics['distraction'] = 'Distracted' if distraction_ratio > 0.5 else 'Focused'

                    # Create the feature vector
                    feature_vector = [
                        features['avg_ear'], 
                        features['avg_mar'], 
                        features['avg_pitch'], 
                        features['avg_roll'], 
                        features['avg_yaw'], 
                        drowsiness_ratio, 
                        distraction_ratio
                    ]
                    self.feature_buffer.append(feature_vector)

                    # Predict the attention score when the buffer is full (5 samples)
                    if len(self.feature_buffer) == 5:
                        sequence = np.array(list(self.feature_buffer))

                        # Check the shape before reshaping
                        print("Shape before reshaping:", sequence.shape)  # Debugging line

                        # Ensure the shape matches what we expect
                        if sequence.shape[0] == 5 and sequence.shape[1] == 7:
                            sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])  # Shape should be (5, 7)
                            sequence_scaled = self.scaler.fit_transform(sequence_reshaped)
                            sequence = sequence_scaled.reshape(1, 5, 7)  # Reshape for LSTM input

                            # Make the prediction
                            prediction = self.lstm_model.predict(sequence, verbose=0)
                            attention_score = prediction[0][0]
                            self.metrics['attention_score'] = attention_score  # Update attention score
                            self.feature_buffer.clear()  # Clear the buffer after prediction
                        else:
                            print(f"Unexpected sequence shape: {sequence.shape}")  # Debugging line

                # Update attentiveness based on the current ear_avg and mar
                self.update_attentiveness(ear_avg, mar)

            else:
                self.metrics['face_status'] = 'No Face Detected'
        except Exception as e:
            print(f"Frame processing error: {e}")

    def update_attentiveness(self, ear_avg, mar):
        """Update attentiveness status based on EAR and MAR values."""
        if ear_avg < 0.3 and mar > 0.6:  # Inattentive
            self.metrics['attentiveness'] = 'Inattentive'
        elif ear_avg >= 0.3 and mar <= 0.6:
            self.metrics['attentiveness'] = 'Attentive'
        else:
            self.metrics['attentiveness'] = 'Caution'

    def draw_metrics(self, frame):
        # Draw background for metrics
        cv2.rectangle(frame, (10, 10), (400, 230), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 230), (255, 255, 255), 1)
        
        y = 35
        for metric, value in self.metrics.items():
            if metric == 'face_status':
                color = (0, 255, 0) if value == 'Face Detected' else (0, 0, 255)
            elif metric == 'attentiveness':
                color = (0, 255, 0) if value == 'Attentive' else (0, 0, 255) if value == 'Inattentive' else (0, 255, 255)
            elif metric == 'drowsiness':
                color = (0, 255, 0) if value == 'Alert' else (0, 0, 255)
            elif metric == 'distraction':
                color = (0, 255, 0) if value == 'Focused' else (0, 0, 255)
            else:
                color = (255, 255, 255)

            cv2.putText(frame, f"{metric.capitalize()}: {value}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 30

    def run(self):
        cap = cv2.VideoCapture(0)  # Use your camera source here
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.process_frame(frame)
            self.draw_metrics(frame)
            cv2.imshow('Attention Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    lstm_model_path = r"C:\Users\DELL\summerInternship\data\trained_models\lstm_model.h5"
    emotion_model_path = r"C:\Users\DELL\summerInternship\data\trained_models\emotion_model.h5"
    monitor = RealtimeAttentionMonitor(lstm_model_path, emotion_model_path)
    monitor.run()