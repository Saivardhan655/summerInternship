import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

class Config:
    SEQUENCE_LENGTH = 5
    BATCH_SIZE = 16
    EPOCHS = 20
    LSTM_UNITS = 64
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 0.001
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
class AttentivenessTrainer:
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.emotion_encoder = LabelEncoder()
        
    def prepare_sequences(self, features_df):
        """Convert features into sequences for LSTM training with improved feature handling"""
        sequences = []
        labels = []
        
        try:
            # Encode emotional features
            features_df['dominant_emotion_encoded'] = self.emotion_encoder.fit_transform(features_df['dominant_emotion'])
            
            # Define feature columns
            numerical_features = [
                'avg_ear', 'avg_mar', 'avg_pitch', 'avg_roll', 'avg_yaw',
                'emotion_diversity', 'drowsiness_ratio', 'distraction_ratio'
            ]
            
            # Verify all required columns exist
            missing_columns = [col for col in numerical_features if col not in features_df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns in dataset: {missing_columns}")
            
            for video_file in features_df['video_file'].unique():
                video_data = features_df[features_df['video_file'] == video_file].sort_values('segment_id')
                
                # Combine numerical and encoded categorical features
                X = np.column_stack([
                    video_data[numerical_features].values,
                    video_data['dominant_emotion_encoded'].values.reshape(-1, 1)
                ])
                
                # Scale features
                if len(X) > 0:
                    X = self.feature_scaler.fit_transform(X)
                
                # Create sequences with overlap
                for i in range(len(X) - self.config.SEQUENCE_LENGTH + 1):
                    sequence = X[i:(i + self.config.SEQUENCE_LENGTH)]
                    sequences.append(sequence)
                    labels.append(self._calculate_attentiveness_label(
                        video_data.iloc[i:(i + self.config.SEQUENCE_LENGTH)]
                    ))
            
            if not sequences:
                raise ValueError("No valid sequences could be created from the input data")
                
            return np.array(sequences), np.array(labels)
            
        except Exception as e:
            print(f"Error in prepare_sequences: {str(e)}")
            raise
    
    def _calculate_attentiveness_label(self, segment_data):
        """
        Enhanced attentiveness calculation using multiple features
        Returns 1 for attentive, 0 for inattentive
        """
        try:
            # Weight different factors
            drowsiness_weight = 0.4
            distraction_weight = 0.3
            ear_weight = 0.2
            emotion_weight = 0.1
            
            # Calculate component scores
            drowsiness_score = 1 - segment_data['drowsiness_ratio'].mean()
            distraction_score = 1 - segment_data['distraction_ratio'].mean()
            ear_score = (segment_data['avg_ear'].mean() - 0.2) / 0.15  # Normalize EAR
            
            # Consider emotional state (assuming negative emotions indicate less attentiveness)
            negative_emotions = ['Sad', 'Angry', 'Fearful']
            emotion_score = 1 - (segment_data['dominant_emotion'].isin(negative_emotions).mean())
            
            # Calculate weighted average
            total_score = (
                drowsiness_weight * drowsiness_score +
                distraction_weight * distraction_score +
                ear_weight * ear_score +
                emotion_weight * emotion_score
            )
            
            # Use threshold for classification
            return 1 if total_score > 0.6 else 0
            
        except Exception as e:
            print(f"Error in _calculate_attentiveness_label: {str(e)}")
            raise
    
    def build_model(self, n_features):
        """Create an enhanced LSTM model architecture"""
        model = Sequential([
            # Bidirectional LSTM layers
            Bidirectional(LSTM(self.config.LSTM_UNITS, return_sequences=True), 
                         input_shape=(self.config.SEQUENCE_LENGTH, n_features)),
            BatchNormalization(),
            Dropout(self.config.DROPOUT_RATE),
            
            Bidirectional(LSTM(self.config.LSTM_UNITS // 2)),
            BatchNormalization(),
            Dropout(self.config.DROPOUT_RATE),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.config.DROPOUT_RATE/2),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Use Adam optimizer with custom learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train_model(self, features_csv):
        """Train the enhanced LSTM model with additional callbacks"""
        try:
            print("Loading and preprocessing data...")
            if not os.path.exists(features_csv):
                raise FileNotFoundError(f"Features file not found: {features_csv}")
                
            features_df = pd.read_csv(features_csv)
            
            print("Preparing sequences...")
            X, y = self.prepare_sequences(features_df)
            
            print("Splitting dataset...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("Building model...")
            model = self.build_model(n_features=X_train.shape[2])
            
            # Create model directory if it doesn't exist
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            
            # Setup callbacks with .keras extension
            model_path = os.path.join(self.config.MODEL_DIR, 'best_model.keras')
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            print("Training model...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                callbacks=callbacks,
                class_weight={0: 1.0, 1: 1.0},  # Adjust if classes are imbalanced
                verbose=1
            )
            
            return model, history
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            raise

def plot_training_history(history):
    """Plot training metrics"""
    try:
        import matplotlib.pyplot as plt
        
        metrics = ['loss', 'accuracy', 'auc']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, metric in enumerate(metrics):
            axes[idx].plot(history.history[metric], label=f'Training {metric}')
            axes[idx].plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            axes[idx].set_title(f'Model {metric.capitalize()}')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in plot_training_history: {str(e)}")

def main():
    try:
        # Initialize trainer
        config = Config()
        trainer = AttentivenessTrainer(config)
        
        # Get absolute path for features.csv
        features_csv = os.path.abspath('C:/Users/DELL/summerInternship/data/processed/features.csv')
        
        # Train model
        model, history = trainer.train_model(features_csv)
        
        # Plot training history
        plot_training_history(history)
        
        # Print model summary
        model.summary()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()