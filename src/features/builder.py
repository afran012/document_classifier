# src/features/builder.py

import tensorflow as tf
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet'
        )
    
    def preprocess_image(self, image):
        """
        Preprocesar imagen para el modelo
        """
        # Normalización y redimensionamiento
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        return image
    
    def extract_features(self, image):
        """
        Extraer características usando ResNet50
        """
        preprocessed = self.preprocess_image(image)
        features = self.base_model.predict(np.array([preprocessed]))
        return features