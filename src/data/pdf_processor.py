from pdf2image import convert_from_path
import os
import json
from datetime import datetime
import logging
import cv2
import numpy as np
from pyspark.sql.types import *
from pyspark.sql.functions import udf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, spark=None, raw_dir="data/raw", processed_dir="data/processed"):
        self.spark = spark
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def process_pdfs(self):
        """Convierte todos los PDFs en el directorio raw a imágenes"""
        for pdf_file in os.listdir(self.raw_dir):
            if pdf_file.endswith('.pdf'):
                self.process_pdf(pdf_file)

    def process_pdf(self, pdf_filename):
        """
        Procesa un solo PDF y retorna información de sus páginas
        """
        pdf_path = os.path.join(self.raw_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            logger.error(f"Error: No se encuentra el archivo {pdf_path}")
            return None

        # Crear directorio para este PDF
        pdf_name = os.path.splitext(pdf_filename)[0]
        pdf_output_dir = os.path.join(self.processed_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        # Convertir PDF a imágenes
        logger.info(f"Procesando {pdf_filename}...")
        pages = convert_from_path(pdf_path)
        
        # Guardar información de páginas
        page_info = []
        for i, page in enumerate(pages):
            page_path = os.path.join(pdf_output_dir, f'page_{i}.png')
            page.save(page_path)
            page_info.append({
                'page_number': i,
                'image_path': page_path
            })

        pdf_info = {
            'pdf_name': pdf_filename,
            'total_pages': len(pages),
            'pages': page_info,
            'processed_date': datetime.now().isoformat()
        }

        self.save_pdf_info(pdf_info)
        return pdf_info

    def save_pdf_info(self, pdf_info, output_file='labels/pdf_info.json'):
        """
        Guarda la información del PDF procesado
        """
        os.makedirs('labels', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(pdf_info, f, indent=4)
        logger.info(f"Información del PDF guardada en {output_file}")

    def create_spark_dataframe(self, processed_pages, labels=None):
        """
        Crea un DataFrame de Spark con las páginas procesadas
        """
        if not self.spark:
            logger.error("Spark session no proporcionada")
            return None

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
        for pdf_file in os.listdir(self.raw_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.raw_dir, pdf_file)
                logger.info(f"Procesando {pdf_file}")
                pages = self.process_pdf(pdf_path)
                all_pages.extend(pages)

        return self.create_spark_dataframe(all_pages)

if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_pdfs()