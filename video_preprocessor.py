import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector

class VideoPreprocessor:
    def __init__(self):
        self.face_detector = FaceDetector('data/trained_models/shape_predictor_68_face_landmarks.dat')
        self.drowsiness_detector = DrowsinessDetector(ear_threshold=0.3, mar_threshold=0.6)
        self.distraction_detector = DistractionDetector(pose_threshold=30)
        # Load pre-trained emotion model
        self.emotion_model = load_model('data/trained_models/emotion_model.h5')
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    def split_video(self, video_path, output_dir, segment_length=2):
        """Split video into 2-second segments"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames_per_segment = fps * segment_length
        
        frame_count = 0
        segment_frames = []
        segment_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            segment_frames.append(frame)
            frame_count += 1
            
            if frame_count == frames_per_segment:
                self._save_segment(segment_frames, output_dir, segment_number)
                segment_frames = []
                frame_count = 0
                segment_number += 1
        
        cap.release()
        return segment_number
    
    def _save_segment(self, frames, output_dir, segment_number):
        """Save video segment"""
        if not frames:
            return
        
        output_path = os.path.join(output_dir, f'segment_{segment_number:04d}.mp4')
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
    
    def extract_features(self, video_path):
        """Extract all features from a video segment"""
        cap = cv2.VideoCapture(video_path)
        features = {
            'ear_values': [],
            'mar_values': [],
            'pitch': [],
            'roll': [],
            'yaw': [],
            'emotions': []
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face and landmarks
            face, landmarks = self.face_detector.detect(frame)
            if face is None:
                continue
            
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
            
            # Extract emotion features
            face_roi = self._extract_face_roi(frame, face)
            emotion_pred = self.emotion_model.predict(face_roi)
            emotion_label = self.emotions[np.argmax(emotion_pred)]
            
            # Store features
            features['ear_values'].append(ear_avg)
            features['mar_values'].append(mar)
            features['pitch'].append(euler_angles[0])
            features['roll'].append(euler_angles[1])
            features['yaw'].append(euler_angles[2])
            features['emotions'].append(emotion_label)
        
        cap.release()
        return self._aggregate_features(features)
    
    def _extract_face_roi(self, frame, face):
        """Extract and preprocess face ROI for emotion detection"""
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)
        return face_roi
    
    def _aggregate_features(self, features):
        """Aggregate features over the video segment"""
        emotion_counts = pd.Series(features['emotions']).value_counts()
        dominant_emotion = emotion_counts.index[0]
        if emotion_counts.empty:
            dominant_emotion = 'Unknown'
        else:
            dominant_emotion = emotion_counts.index[0]
        return {
            'avg_ear': np.mean(features['ear_values']),
            'avg_mar': np.mean(features['mar_values']),
            'avg_pitch': np.mean(features['pitch']),
            'avg_roll': np.mean(features['roll']),
            'avg_yaw': np.mean(features['yaw']),
            'dominant_emotion': dominant_emotion,
            'emotion_diversity': len(emotion_counts),
            'drowsiness_ratio': np.mean(np.array(features['ear_values']) < 0.3),
            'distraction_ratio': np.mean(np.abs(features['yaw']) > 30)
        }

def process_dataset(video_dir, output_dir):
    """Process entire dataset and create features CSV"""
    preprocessor = VideoPreprocessor()
    all_features = []
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue
            
        video_path = os.path.join(video_dir, video_file)
        segments_dir = os.path.join(output_dir, 'segments', video_file[:-4])
        os.makedirs(segments_dir, exist_ok=True)
        num_segments = preprocessor.split_video(video_path, segments_dir)

        for i in range(num_segments):
            segment_path = os.path.join(segments_dir, f'segment_{i:04d}.mp4')
            features = preprocessor.extract_features(segment_path)
            features['video_file'] = video_file
            features['segment_id'] = i
            all_features.append(features)
    
    # Create features DataFrame
    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    return df

if __name__ == "__main__":
    video_dir = "data/videos/raw"
    output_dir = "data/processed"
    features_df = process_dataset(video_dir, output_dir)