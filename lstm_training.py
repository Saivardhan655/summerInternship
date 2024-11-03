import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.lstm_model import AttentivenessLSTM  # Assuming this model has been defined in your codebase
from config import Config  # Configuration settings

class AttentivenessTrainer:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def prepare_sequences(self, features_df):
        """Convert features into sequences for LSTM training"""
        sequences = []
        labels = []
        
        for video_file in features_df['video_file'].unique():
            video_data = features_df[features_df['video_file'] == video_file].sort_values('segment_id')
            
            # Extract features
            feature_cols = ['avg_ear', 'avg_mar', 'avg_pitch', 'avg_roll', 'avg_yaw',
                            'drowsiness_ratio', 'distraction_ratio']
            X = video_data[feature_cols].values
            
            # Scale features
            if len(X) > 0:
                X = self.scaler.fit_transform(X)
            
            # Create sequences
            for i in range(len(X) - self.config.SEQUENCE_LENGTH):
                sequences.append(X[i:(i + self.config.SEQUENCE_LENGTH)])
                labels.append(self._calculate_attentiveness_label(
                    video_data.iloc[i:(i + self.config.SEQUENCE_LENGTH)]
                ))
        
        return np.array(sequences), np.array(labels)
    
    def _calculate_attentiveness_label(self, segment_data):
        """
        Calculate attentiveness label based on features
        Returns 1 for attentive, 0 for inattentive
        Modify this based on your specific criteria
        """
        avg_drowsiness = segment_data['drowsiness_ratio'].mean()
        avg_distraction = segment_data['distraction_ratio'].mean()
        
        # Adjust threshold values based on your requirements
        is_attentive = (avg_drowsiness < 0.3) and (avg_distraction < 0.3)
        
        return 1 if is_attentive else 0
    
    def train_model(self, features_csv):
        """Train the LSTM model"""
        print("Loading features from CSV...")
        features_df = pd.read_csv(features_csv)
        
        print("Preparing sequences...")
        X, y = self.prepare_sequences(features_df)
        
        print("Splitting dataset...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Initializing model...")
        n_features = X_train.shape[2]
        model = AttentivenessLSTM(
            sequence_length=self.config.SEQUENCE_LENGTH,
            n_features=n_features
        )
        
        print("Training model...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS
        )
        
        print("Saving model...")
        model_path = os.path.join(self.config.MODEL_DIR, 'lstm_model.h5')
        model.save(model_path)
        
        return history

def main():
    config = Config()
    trainer = AttentivenessTrainer(config)
    
    features_csv = os.path.join(config.BASE_DIR, 'data', 'processed', 'features.csv')
    
    if not os.path.exists(features_csv):
        print("Processing video dataset to extract features...")
        from video_preprocessor import process_dataset
        process_dataset(config.VIDEO_DIR, os.path.dirname(features_csv))
    
    # Train the model
    history = trainer.train_model(features_csv)
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()