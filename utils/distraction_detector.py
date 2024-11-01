import cv2
import numpy as np

class DistractionDetector:
    def __init__(self, pose_threshold):
        self.pose_threshold = pose_threshold
        
    def get_head_pose(self, landmarks):
        """Calculate head pose estimation using 6 facial landmarks"""
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D points
        image_points = np.array([
            landmarks[30],    # Nose tip
            landmarks[8],     # Chin
            landmarks[36],    # Left eye corner
            landmarks[45],    # Right eye corner
            landmarks[48],    # Left mouth corner
            landmarks[54]     # Right mouth corner
        ], dtype=float)
        
        # Camera matrix
        focal_length = 500
        center = (450, 300)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=float)
        
        # Distortion coefficients
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs
        )
        
        # Convert rotation vector to rotation matrix and euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        return euler_angles
    
    def _rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to euler angles"""
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z]) * 180.0 / np.pi
    
    def is_distracted(self, landmarks):
        euler_angles = self.get_head_pose(landmarks)
        return np.any(np.abs(euler_angles) > self.pose_threshold)