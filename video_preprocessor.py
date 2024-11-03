#video_preprocessor.py
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
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_path}")
            return None
            
        features = {
            'ear_values': [],
            'mar_values': [],
            'pitch': [],
            'roll': [],
            'yaw': [],
            'emotions': []
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            try:
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
                emotion_pred = self.emotion_model.predict(face_roi, verbose=0)  # Added verbose=0 to reduce output
                emotion_label = self.emotions[np.argmax(emotion_pred)]
                
                # Store features
                features['ear_values'].append(ear_avg)
                features['mar_values'].append(mar)
                features['pitch'].append(euler_angles[0])
                features['roll'].append(euler_angles[1])
                features['yaw'].append(euler_angles[2])
                features['emotions'].append(emotion_label)
                
            except Exception as e:
                print(f"Warning: Error processing frame {frame_count} in {video_path}: {str(e)}")
                continue
        
        cap.release()
        
        if frame_count == 0:
            print(f"Warning: No frames processed in {video_path}")
            return None
            
        if not features['emotions']:
            print(f"Warning: No faces detected in {video_path}")
            
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
        # Handle empty features
        if not features['emotions']:
            return {
                'avg_ear': 0.0,
                'avg_mar': 0.0,
                'avg_pitch': 0.0,
                'avg_roll': 0.0,
                'avg_yaw': 0.0,
                'dominant_emotion': 'Unknown',
                'emotion_diversity': 0,
                'drowsiness_ratio': 0.0,
                'distraction_ratio': 0.0
            }
            
        # Calculate emotion statistics
        emotion_counts = pd.Series(features['emotions']).value_counts()
        
        # Safely get statistics
        ear_values = features['ear_values'] or [0.0]
        mar_values = features['mar_values'] or [0.0]
        pitch_values = features['pitch'] or [0.0]
        roll_values = features['roll'] or [0.0]
        yaw_values = features['yaw'] or [0.0]
        
        return {
            'avg_ear': np.mean(ear_values),
            'avg_mar': np.mean(mar_values),
            'avg_pitch': np.mean(pitch_values),
            'avg_roll': np.mean(roll_values),
            'avg_yaw': np.mean(yaw_values),
            'dominant_emotion': emotion_counts.index[0] if not emotion_counts.empty else 'Unknown',
            'emotion_diversity': len(emotion_counts),
            'drowsiness_ratio': np.mean(np.array(ear_values) < 0.3),
            'distraction_ratio': np.mean(np.abs(yaw_values) > 30)
        }

def process_dataset(video_dir, output_dir):
    """Process entire dataset and create features CSV"""
    preprocessor = VideoPreprocessor()
    all_features = []
    
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith('.mp4'):
            continue
            
        video_path = os.path.join(video_dir, video_file)
        segments_dir = os.path.join(output_dir, 'segments', video_file[:-4])
        os.makedirs(segments_dir, exist_ok=True)
        
        try:
            num_segments = preprocessor.split_video(video_path, segments_dir)
            print(f"Processing {video_file}: {num_segments} segments created")
            
            for i in range(num_segments):
                segment_path = os.path.join(segments_dir, f'segment_{i:04d}.mp4')
                features = preprocessor.extract_features(segment_path)
                
                if features is not None:
                    features['video_file'] = video_file
                    features['segment_id'] = i
                    all_features.append(features)
                    
        except Exception as e:
            print(f"Error processing video {video_file}: {str(e)}")
            continue
    
    if not all_features:
        print("Warning: No features extracted from any video")
        return pd.DataFrame()
        
    # Create features DataFrame
    new_df = pd.DataFrame(all_features)
    output_file = os.path.join(output_dir, 'features.csv')
    
    try:
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")
    except Exception as e:
        print(f"Error saving features to CSV: {str(e)}")
        return new_df
        
    return combined_df

if __name__ == "__main__":
    video_dir = "data/videos/raw"
    output_dir = "data/processed"
    features_df = process_dataset(video_dir, output_dir)