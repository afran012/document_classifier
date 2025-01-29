import os
from pdf2image import convert_from_path
import json
from datetime import datetime

class PDFProcessor:
    def __init__(self, pdf_dir="data/raw_pdfs", output_dir="data/processed"):
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_pdf(self, pdf_filename):
        """
        Procesa un solo PDF y retorna información de sus páginas
        """
        pdf_path = os.path.join(self.pdf_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"Error: No se encuentra el archivo {pdf_path}")
            return None

        # Crear directorio para este PDF
        pdf_name = os.path.splitext(pdf_filename)[0]
        pdf_output_dir = os.path.join(self.output_dir, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        # Convertir PDF a imágenes
        print(f"Procesando {pdf_filename}...")
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

        return {
            'pdf_name': pdf_filename,
            'total_pages': len(pages),
            'pages': page_info,
            'processed_date': datetime.now().isoformat()
        }

    def save_pdf_info(self, pdf_info, output_file='labels/pdf_info.json'):
        """
        Guarda la información del PDF procesado
        """
        os.makedirs('labels', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(pdf_info, f, indent=4)