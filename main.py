import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as LE
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
# Load iris dataset
data = load_iris()
X = data.data
y = data.target
# One-hot encode the target
y_encoded = to_categorical(y)
# Train-test split
X_train, X_test, y_train, y_test = tts(X, y_encoded, test_size=0.2, random_state=42)
model = Seq()
# Input layer and first hidden layer
model.add(Dense(10, input_shape=(4,), activation='relu'))

# Second hidden layer
model.add(Dense(8, activation='relu'))

# Output layer (3 classes for Iris)
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
