from pdf2image import convert_from_path
import os
import cv2
import numpy as np
import logging

class PDFProcessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def process_pdfs(self):
        """Convierte PDFs a imágenes"""
        for pdf_file in os.listdir(self.raw_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(self.raw_dir, pdf_file)
                pdf_name = pdf_file.replace('.pdf', '')
                
                # Crear directorio para este PDF
                pdf_dir = os.path.join(self.processed_dir, pdf_name)
                os.makedirs(pdf_dir, exist_ok=True)
                
                # Convertir PDF a imágenes
                pages = convert_from_path(pdf_path)
                
                # Guardar cada página
                for i, page in enumerate(pages):
                    page_path = os.path.join(pdf_dir, f'page_{i}.png')
                    page.save(page_path)