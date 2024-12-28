import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# File paths
MODEL_PATH = 'models/genre_classifier.h5'
FEEDBACK_FILE = 'feedback_log.txt'

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Define genres
GENRES = ['Pop', 'Rock', 'Jazz', 'Classical', 'Hip-hop']

# Load or create the model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print("Model not found. Creating a new model...")
    # Modify the input shape of the model
    input_shape = 40  # Change this to 40 to match the MFCCs you want to use

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(GENRES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Define allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess audio
def preprocess_audio(file):
    print(f"Preprocessing file: {file.filename}")
    
    # Load the audio file
    y_audio, sr = librosa.load(file, mono=True, duration=30)
    print(f"Audio shape: {y_audio.shape} Sample rate: {sr}")
    
    # Extract MFCCs (now set to 100 to match the model's input shape)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=100)  # Increase to 100 MFCCs
    print(f"MFCC Shape: {mfcc.shape}")  # Print the shape of MFCC features
    
    # Scale MFCCs to match the training format (mean of each MFCC coefficient)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    print(f"Scaled MFCC: {mfcc_scaled[:10]}...")  # Print first 10 elements to check
    
    # Return the extracted features
    return mfcc_scaled


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # Check if the file has a valid extension
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format. Only audio files are allowed."}), 400
    
    # Simulate audio preprocessing and feature extraction
    features = preprocess_audio(file)
    
    # Ensure the correct shape for the model input
    features = np.array(features).reshape(1, -1)  # Add batch dimension
    print(f"Input features shape: {features.shape}")  # Ensure the input has the right shape
    
    # Predict genre
    predictions = model.predict(features)
    print(f"Predictions: {predictions}")  # Check raw model output (before softmax)
    
    predicted_genre = GENRES[predictions.argmax()]  # Get the predicted genre
    print(f"Predicted Genre: {predicted_genre}")
    
    return jsonify({"genre": predicted_genre})

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        # Get feedback data from the request
        feedback_data = request.get_json()
        correct = feedback_data['correct']
        genre = feedback_data['genre']
        
        # Log feedback to a file (or you can store it in a database)
        with open(FEEDBACK_FILE, 'a') as f:
            f.write(f"Genre: {genre}, Correct: {correct}\n")
        
        print(f"Feedback received: Genre = {genre}, Correct = {correct}")
        
        return jsonify({"message": "Feedback received"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process feedback: {str(e)}"}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
