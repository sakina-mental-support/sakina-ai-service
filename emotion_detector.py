import os
import pickle
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class EmotionDetector:
    def __init__(self):
        # Load the trained model and processor
        try:
            self.model = tf.keras.models.load_model('models/emotion_model.h5')
            with open('models/processor.pkl', 'rb') as f:
                self.processor = pickle.load(f)
            print("Model and processor loaded successfully.")
        except Exception as e:
            print(f"Error loading models (have you run train_model.py?): {e}")
            self.model = None
            self.processor = None
            
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    def predict_emotion(self, audio_file_path):
        if not self.model or not self.processor:
            return "Model not loaded properly"
            
        # extract features using our previously defined processor
        features = self.processor.extract_features(audio_file_path)
        if features is None:
            return "Error extracting features"
            
        # The model's prediction expects shape (batch_size, num_features)
        # So we add an extra dimension for batch=1
        features = np.expand_dims(features, axis=0)
        predictions = self.model.predict(features)
        
        # get the index of the highest probability
        predicted_idx = np.argmax(predictions[0])
        
        # map index back to the emotion string using the LabelEncoder
        predicted_label = self.processor.le.inverse_transform([predicted_idx])[0]
        return predicted_label
        
    def generate_therapeutic_response(self, emotion):
        if not os.getenv("GEMINI_API_KEY"):
            return "Gemini API key not found. Please add it to your .env file."
            
        prompt = f"""
        Act as an empathetic, supportive AI therapist.
        The user has spoken, and their detected emotion from their voice is '{emotion}'.
        Please provide a short, validating, and calming response tailored to someone feeling {emotion}.
        Keep the response concise (2-3 sentences max) and use a gentle, professional tone.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
