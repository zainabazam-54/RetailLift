import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2


class ShopliftingClassifier:
    def __init__(self, model_path, threshold=0.5):
        self.model = keras.models.load_model(model_path)
        self.threshold = threshold

    def preprocess_frame(self, img, resize=(224, 224)):
        img = cv2.resize(img, resize)
        img = img / 255.0
        return img

    def predict(self, person_crop_sequence):
        """
        person_crop_sequence: list of np.array frames of shape (H,W,C)
        """
        if len(person_crop_sequence) == 0:
            return "Normal", 0.0

        # Convert all frames to array and stack
        frames = np.array([self.preprocess_frame(f) for f in person_crop_sequence])
        
        # Add batch dimension if needed
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=0)

        # Get prediction
        output = self.model.predict(frames, verbose=0)
        prob = float(output[0][0]) if output.shape[-1] == 1 else float(output[0][1])
        label = "Shoplifting" if prob > self.threshold else "Normal"

        return label, prob
