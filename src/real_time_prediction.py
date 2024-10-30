import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
from feature_extraction import calculate_ear, calculate_mar

# Load trained models
lstm_model = load_model("../models/lstm_model.h5")
emotion_model = load_model("../models/emotion_model.h5")
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def real_time_prediction():
    cap = cv2.VideoCapture(0)
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Calculate EAR and MAR
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth)

            face_crop = gray[face.top():face.bottom(), face.left():face.right()]
            face_resized = cv2.resize(face_crop, (48, 48))
            face_resized = np.expand_dims(face_resized, axis=[0, -1])
            emotion = emotion_model.predict(face_resized)[0]

            frame_features = [ear, mar, *emotion]
            frame_buffer.append(frame_features)

            if len(frame_buffer) == 49:
                pred = lstm_model.predict(np.expand_dims(frame_buffer, axis=0))
                focus_status = "Focused" if pred > 0.5 else "Not Focused"
                print(f"Focus Status: {focus_status}")
                frame_buffer.pop(0)

        cv2.imshow("Real-time Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_prediction()
