import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load labeled features
data = pd.read_csv("../data/labeled_features.csv")

# Prepare data for LSTM
X = np.array([data.iloc[i:i+49].values for i in range(len(data) - 49)])
y = np.array([1 if data.iloc[i+49]["label"] == "Focused" else 0 for i in range(len(data) - 49)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build and train LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(49, X.shape[2])),
    LSTM(50),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save("../models/lstm_model.h5")
