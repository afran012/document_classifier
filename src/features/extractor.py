import tensorflow as tf
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import cv2
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.base_model = self._create_base_model()

    def _create_base_model(self):
        """
        Crea el modelo base para extracción de características
        """
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        base_model.trainable = False
        return base_model

    def extract_features(self, img_array):
        """
        Extrae características de una imagen
        """
        try:
            # Convertir imagen en escala de grises a RGB
            if len(img_array.shape) == 2 or img_array.shape[-1] == 1:
                img_rgb = np.stack([img_array] * 3, axis=-1)
            else:
                img_rgb = img_array

            # Preprocesamiento para ResNet
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = tf.keras.applications.resnet_v2.preprocess_input(img_resized)
            
            # Extraer características
            features = self.base_model.predict(np.expand_dims(img_normalized, axis=0))
            return features.flatten()

        except Exception as e:
            logger.error(f"Error en extracción de características: {str(e)}")
            return None

    def create_spark_udf(self):
        """
        Crea UDF de Spark para extracción de características
        """
        def extract_features_udf(img_path):
            try:
                img = cv2.imread(img_path)
                features = self.extract_features(img)
                return features.tolist() if features is not None else None
            except Exception as e:
                logger.error(f"Error en UDF: {str(e)}")
                return None

        return udf(extract_features_udf, ArrayType(FloatType()))

    def extract_batch_features(self, image_paths):
        """
        Extrae características de un lote de imágenes
        """
        features_list = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                features = self.extract_features(img)
                if features is not None:
                    features_list.append(features)
            except Exception as e:
                logger.error(f"Error procesando {path}: {str(e)}")

        return np.array(features_list)