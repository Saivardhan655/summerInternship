import sys
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

class AttentivenessPredictor:
    def __init__(self, model_path='best_model.keras'):
        self.model = load_model(model_path)
    
    def preprocess_frame(self, frame_data):
        # Decode base64 frame
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize and preprocess
        resized = cv2.resize(frame, (224, 224))
        preprocessed = preprocess_input(np.expand_dims(resized, axis=0))
        
        return preprocessed
    
    def predict_attentiveness(self, frame_data):
        processed_frame = self.preprocess_frame(frame_data)
        prediction = self.model.predict(processed_frame)
        return float(prediction[0][0])

def main():
    # Read input from command line
    frame_data = sys.argv[1]
    user_id = sys.argv[2]
    room_id = sys.argv[3]
    
    predictor = AttentivenessPredictor()
    attentiveness = predictor.predict_attentiveness(frame_data)
    
    # Output result as JSON
    print(json.dumps({
        'userId': user_id,
        'roomId': room_id,
        'attentiveness': attentiveness
    }))

if __name__ == '__main__':
    main()