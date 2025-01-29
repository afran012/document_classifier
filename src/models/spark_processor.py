from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import numpy as np
import cv2
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkImageProcessor:
    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName("DocumentClassifier") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()

    def _create_image_schema(self):
        """
        Crea el esquema para las imágenes
        """
        return StructType([
            StructField("path", StringType(), True),
            StructField("image", BinaryType(), True),
            StructField("label", IntegerType(), True)
        ])

    def preprocess_image_udf(self):
        """
        UDF para preprocesar imágenes
        """
        def preprocess(image_array):
            try:
                # Convertir bytes a numpy array
                nparr = np.frombuffer(image_array, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                # Redimensionar
                resized = cv2.resize(img, (224, 224))
                
                # Normalizar
                normalized = resized / 255.0
                
                return normalized.tolist()
            except Exception as e:
                logger.error(f"Error en preprocesamiento: {str(e)}")
                return None
        
        return udf(preprocess, ArrayType(ArrayType(FloatType())))

    def process_images(self, image_paths: List[str], labels: List[int] = None):
        """
        Procesa un conjunto de imágenes usando Spark
        """
        # Crear DataFrame con paths e imágenes
        if labels is None:
            labels = [0] * len(image_paths)

        data = [(path, path, label) for path, label in zip(image_paths, labels)]
        df = self.spark.createDataFrame(data, ["path", "image", "label"])

        # Aplicar preprocesamiento
        preprocess_udf = self.preprocess_image_udf()
        processed_df = df.withColumn("processed_image", preprocess_udf("image"))

        return processed_df

    def create_pipeline(self):
        """
        Crea un pipeline de procesamiento
        """
        assembler = VectorAssembler(
            inputCols=["processed_image"],
            outputCol="features"
        )

        pipeline = Pipeline(stages=[
            assembler
        ])

        return pipeline

    def stop_spark(self):
        """
        Detiene la sesión de Spark
        """
        if self.spark:
            self.spark.stop()