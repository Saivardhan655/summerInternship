import dlib
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
    
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None, None
        
        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = self._shape_to_np(landmarks)
        
        return face, landmarks
    
    def _shape_to_np(self, shape):
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords