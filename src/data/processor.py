import os
from pdf2image import convert_from_path
import cv2
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, spark, input_dir="data/raw", output_dir="data/processed"):
        self.spark = spark
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_pdf(self, pdf_path):
        """
        Procesa un PDF y lo convierte en imágenes
        """
        try:
            # Convertir PDF a imágenes
            pages = convert_from_path(pdf_path)
            
            # Crear directorio para este PDF
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            pdf_dir = os.path.join(self.output_dir, pdf_name)
            os.makedirs(pdf_dir, exist_ok=True)
            
            processed_pages = []
            for i, page in enumerate(pages):
                # Guardar imagen original
                page_path = os.path.join(pdf_dir, f'page_{i}.png')
                page.save(page_path)
                
                # Convertir a array numpy
                img_array = np.array(page)
                processed_pages.append({
                    'page_number': i,
                    'path': page_path,
                    'array': img_array
                })
            
            return processed_pages
        except Exception as e:
            logger.error(f"Error procesando PDF {pdf_path}: {str(e)}")
            return []

    def create_spark_dataframe(self, processed_pages, labels=None):
        """
        Crea un DataFrame de Spark con las páginas procesadas
        """
        # Definir esquema
        schema = StructType([
            StructField("page_number", IntegerType(), False),
            StructField("path", StringType(), False),
            StructField("label", IntegerType(), True)
        ])

        # Crear datos para el DataFrame
        data = []
        for page in processed_pages:
            row = {
                "page_number": page['page_number'],
                "path": page['path'],
                "label": labels.get(page['page_number'], 0) if labels else 0
            }
            data.append(row)

        return self.spark.createDataFrame(data, schema)

    @staticmethod
    def preprocess_image(image_array, target_size=(224, 224)):
        """
        Preprocesa una imagen para el modelo
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array

            # Redimensionar
            resized = cv2.resize(gray, target_size)

            # Normalizar
            normalized = resized.astype(np.float32) / 255.0

            return normalized[..., np.newaxis]
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {str(e)}")
            return None

    def process_directory(self):
        """
        Procesa todos los PDFs en el directorio de entrada
        """
        all_pages = []
        for pdf_file in os.listdir(self.input_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.input_dir, pdf_file)
                logger.info(f"Procesando {pdf_file}")
                pages = self.process_pdf(pdf_path)
                all_pages.extend(pages)

        return self.create_spark_dataframe(all_pages)