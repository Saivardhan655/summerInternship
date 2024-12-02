import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'data', 'trained_models')
    VIDEO_DIR = os.path.join(BASE_DIR, 'data', 'videos')
    
    # Model Parameters
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 16
    EPOCHS = 20
    
    
    # Detection Parameters
    EAR_THRESHOLD = 0.3
    MAR_THRESHOLD = 0.6
    HEAD_POSE_THRESHOLD = 30
    
    # Feature Extraction
    FACIAL_LANDMARKS_PATH = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
    
    # Display Parameters
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    
    # Moving Average Window
    MA_WINDOW = 1

    FEATURE_DIM=os.path.join(VIDEO_DIR,'sample_videos')