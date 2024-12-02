import base64
import cv2
import numpy as np
import threading
from kafka import KafkaConsumer
from io import BytesIO
from tensorflow.keras.models import load_model
from utils.face_detector import FaceDetector
from utils.drowsiness_detector import DrowsinessDetector
from utils.distraction_detector import DistractionDetector

class RealtimeAttentionMonitor:
    def __init__(self, lstm_model_path, emotion_model_path, user_id):
        self.user_id = user_id
        self.lstm_model = load_model(lstm_model_path, compile=False)
        self.face_detector = FaceDetector('data/trained_models/shape_predictor_68_face_landmarks.dat')
        self.drowsiness_detector = DrowsinessDetector(ear_threshold=0.3, mar_threshold=0.6)
        self.distraction_detector = DistractionDetector(pose_threshold=30)
        self.emotion_model = load_model(emotion_model_path)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.feature_buffer = deque(maxlen=5)
        self.frame_features = {
            'ear_values': [],
            'mar_values': [],
            'pitch': [],
            'roll': [],
            'yaw': [],
            'emotions': []
        }
        self.metrics = {
            'face_status': 'No Face Detected',
            'attentiveness': 'Unknown',
            'attention_score': 0.0,
            'current_emotion': 'Unknown',
            'drowsiness': 'N.A',
            'distraction': 'N.A.'
        }

    def process_frame(self, frame):
        try:
            face, landmarks = self.face_detector.detect(frame)
            if face is not None:
                self.metrics['face_status'] = 'Face Detected'
                x, y, w, h = face.left(), face.top(), face.height(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                left_eye = landmarks[42:48]
                right_eye = landmarks[36:42]
                mouth = landmarks[48:60]

                ear_left = self.drowsiness_detector.calculate_ear(left_eye)
                ear_right = self.drowsiness_detector.calculate_ear(right_eye)
                ear_avg = (ear_left + ear_right) / 2
                mar = self.drowsiness_detector.calculate_mar(mouth)

                euler_angles = self.distraction_detector.get_head_pose(landmarks)
                face_roi = self._extract_face_roi(frame, face)
                emotion_pred = self.emotion_model.predict(face_roi, verbose=0)
                emotion_label = self.emotions[np.argmax(emotion_pred)]

                self.frame_features['ear_values'].append(ear_avg)
                self.frame_features['mar_values'].append(mar)
                self.frame_features['pitch'].append(euler_angles[0])
                self.frame_features['roll'].append(euler_angles[1])
                self.frame_features['yaw'].append(euler_angles[2])
                self.frame_features['emotions'].append(emotion_label)

                features = self._aggregate_features()
                if features:
                    self.metrics['current_emotion'] = features['dominant_emotion']
                    drowsiness_ratio = features['drowsiness_ratio']
                    distraction_ratio = features['distraction_ratio']
                    self.metrics['drowsiness'] = 'Drowsy' if drowsiness_ratio > 0.5 else 'Alert'
                    self.metrics['distraction'] = 'Distracted' if distraction_ratio > 0.5 else 'Focused'

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

                    if len(self.feature_buffer) == 5:
                        sequence = np.array(list(self.feature_buffer))
                        if sequence.shape[0] == 5 and sequence.shape[1] == 7:
                            sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
                            sequence_scaled = self.scaler.fit_transform(sequence_reshaped)
                            sequence = sequence_scaled.reshape(1, 5, 7)
                            prediction = self.lstm_model.predict(sequence, verbose=0)
                            attention_score = prediction[0][0]
                            self.metrics['attention_score'] = attention_score * 10
                            self.metrics['attentiveness'] = 'Attentive' if attention_score * 10 >= 0.5 else 'Inattentive'
                            self.feature_buffer.clear()

        except Exception as e:
            print(f"Frame processing error: {e}")

    def _extract_face_roi(self, frame, face):
        x, y, w, h = face.left(), face.top(), face.height(), face.height()
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0)
        return face_roi

    def _aggregate_features(self):
        if not self.frame_features['emotions']:
            return None

        emotion_counts = pd.Series(self.frame_features['emotions']).value_counts()
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
        for key in self.frame_features:
            self.frame_features[key] = []

        return features


def process_user_frames(user_id, base64_image):
    # Decode the base64 image
    img_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Define the paths to the trained models
    lstm_model_path = r"C:\Users\DELL\summerInternship\data\trained_models\lstm_model.h5"
    emotion_model_path = r"C:\Users\DELL\summerInternship\data\trained_models\emotion_model.h5"

    # Initialize and process the frame for the specific user
    monitor = RealtimeAttentionMonitor(lstm_model_path, emotion_model_path, user_id)
    monitor.process_frame(frame)
    monitor.draw_metrics(frame)

    # Display the result (optional)
    cv2.imshow(f"User {user_id} - Attention Monitor", frame)
    cv2.waitKey(1)


def consume_kafka_messages():
    consumer = KafkaConsumer(
        'video_frames_topic',
        bootstrap_servers=['localhost:9092'],
        group_id='attention_monitor_group',
        auto_offset_reset='earliest'
    )

    # Process messages from Kafka
    for message in consumer:
        user_id = message.key.decode('utf-8')
        base64_image = message.value.decode('utf-8')

        # Start a new thread to process the frame for this user
        threading.Thread(target=process_user_frames, args=(user_id, base64_image)).start()


if __name__ == "__main__":
    consume_kafka_messages()
