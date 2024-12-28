import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
# Load dataset (modify path as needed)
DATASET_PATH = 'path_to_dataset/'
GENRES = ['Pop', 'Rock', 'Jazz', 'Classical', 'Hip-hop']

X, y = [], []

for genre_idx, genre in enumerate(GENRES):
    for file in glob.glob(os.path.join(DATASET_PATH, genre, '*.wav')):
        try:
            y_audio, sr = librosa.load(file, mono=True, duration=30)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            X.append(mfcc_scaled)
            y.append(genre_idx)
        except Exception as e:
            print(f"Error loading {file}: {e}")

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(GENRES))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(GENRES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/genre_classifier.h5')