import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Function to extract MFCC features from an audio file
def extract_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)  # Take mean across time axis
    return mfccs_mean

# Load pre-trained model (assuming it's trained on real vs. deepfake audio)
model = joblib.load("deepfake_audio_detector.pkl")

# Function to predict if an audio file is deepfake or real
def predict_audio(audio_path):
    features = extract_features(audio_path).reshape(1, -1)  # Reshape for model input
    prediction = model.predict(features)
    return "Deepfake" if prediction[0] == 1 else "Real"

# Example usage
audio_file = "test_audio.wav"
result = predict_audio(audio_file)
print(f"The audio is classified as: {result}")
