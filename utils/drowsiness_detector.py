import numpy as np

class DrowsinessDetector:
    def __init__(self, ear_threshold, mar_threshold):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        vertical_dist1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_dist2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    
    def calculate_mar(self, mouth_points):
        """Calculate Mouth Aspect Ratio"""
        vertical_dist = np.mean([
            np.linalg.norm(mouth_points[2] - mouth_points[6]),
            np.linalg.norm(mouth_points[3] - mouth_points[5])
        ])
        horizontal_dist = np.linalg.norm(mouth_points[0] - mouth_points[4])
        
        mar = vertical_dist / horizontal_dist
        return mar
    
    def is_drowsy(self, landmarks):
        # Extract eye landmarks
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        
        # Extract mouth landmarks
        mouth = landmarks[48:60]
        
        # Calculate ratios
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        mar = self.calculate_mar(mouth)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        
        return avg_ear < self.ear_threshold or mar > self.mar_threshold