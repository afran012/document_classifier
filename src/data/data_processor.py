# Procesamiento de PDFs e im�genesimport os
from pdf2image import convert_from_path
import cv2
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convierte un PDF a una lista de imágenes
        """
        try:
            images = convert_from_path(pdf_path)
            return [np.array(img) for img in images]
        except Exception as e:
            logger.error(f"Error convirtiendo PDF {pdf_path}: {str(e)}")
            return []

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocesa una imagen para el modelo
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Redimensionar
            resized = cv2.resize(gray, target_size)

            # Normalizar
            normalized = resized / 255.0

            # Aplicar umbral adaptativo
            binary = cv2.adaptiveThreshold(
                (normalized * 255).astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # Reducir ruido
            denoised = cv2.fastNlMeansDenoising(binary)

            # Expandir dimensiones para el modelo
            return np.expand_dims(denoised, axis=-1)

        except Exception as e:
            logger.error(f"Error preprocesando imagen: {str(e)}")
            return None

    def process_pdf_directory(self) -> List[Tuple[str, np.ndarray]]:
        """
        Procesa todos los PDFs en el directorio raw
        """
        processed_images = []

        for pdf_file in os.listdir(self.raw_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.raw_dir, pdf_file)
                logger.info(f"Procesando {pdf_file}")

                # Convertir PDF a imágenes
                images = self.convert_pdf_to_images(pdf_path)

                # Procesar cada imagen
                for i, img in enumerate(images):
                    processed_img = self.preprocess_image(img)
                    if processed_img is not None:
                        # Guardar imagen procesada
                        output_path = os.path.join(
                            self.processed_dir,
                            f"{pdf_file[:-4]}_page_{i}.npy"
                        )
                        np.save(output_path, processed_img)
                        processed_images.append((output_path, processed_img))

        return processed_images

    def get_processed_images(self) -> List[Tuple[str, np.ndarray]]:
        """
        Carga las imágenes procesadas
        """
        processed_images = []
        for file in os.listdir(self.processed_dir):
            if file.endswith('.npy'):
                path = os.path.join(self.processed_dir, file)
                img = np.load(path)
                processed_images.append((path, img))
        return processed_images