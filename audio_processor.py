import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

class AudioEmotionProcessor:
    def __init__(self):
        # We define the 8 emotions present in the RAVDESS dataset.
        # Note: RAVDESS codes are 01 to 08, mapping sequentially to these strings.
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.le = LabelEncoder()
        
    def extract_features(self, file_path):
        """Extracts MFCC, Chroma, and Mel Spectrogram features from an audio file."""
        try:
            # Load audio (3-second duration handles varying length gracefully for fixed shape)
            data, sample_rate = librosa.load(file_path, duration=3, offset=0.5, sr=22050)
            
            # Extract STFT for Chroma
            stft = np.abs(librosa.stft(data))
            
            # MFCCs (40 bands)
            mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
            
            # Chroma STFT
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            
            # Mel Spectrogram
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            
            # Stack all into one 1D array per sample
            features = np.hstack((mfccs, chroma, mel))
            return features
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def prepare_ravdess_dataset(self, path):
        """Processes the RAVDESS dataset from a given path (which contains Actor_01, etc. folders)"""
        X = []
        y = []
        
        # Traverse Actor folders
        for folder in os.listdir(path):
            actor_path = os.path.join(path, folder)
            if not os.path.isdir(actor_path):
                continue
                
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(actor_path, file)
                    # Example filename: 03-01-05-01-01-01-01.wav
                    # The 3rd component (index 2) is the emotion code (01 to 08)
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = int(parts[2]) - 1
                        feature = self.extract_features(file_path)
                        
                        if feature is not None:
                            X.append(feature)
                            # Append the corresponding string label
                            y.append(self.emotions[emotion_code])
                            
        return np.array(X), np.array(y)
