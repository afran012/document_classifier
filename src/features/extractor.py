import tensorflow as tf
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        self.base_model.trainable = False

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características usando ResNet50
        """
        try:
            # Convertir imagen en escala de grises a RGB
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = np.stack((image,) * 3, axis=-1)

            # Asegurar que la imagen esté en el rango [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

            # Expandir dimensiones para el batch
            image = np.expand_dims(image, axis=0)

            # Extraer características
            features = self.base_model.predict(image)
            
            return features.flatten()

        except Exception as e:
            logger.error(f"Error extrayendo características: {str(e)}")
            return None

    def batch_extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extrae características de un lote de imágenes
        """
        features = []
        for img in images:
            feat = self.extract_features(img)
            if feat is not None:
                features.append(feat)
        return np.array(features)