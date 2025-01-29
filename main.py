from src.process_pdf import PDFProcessor
from src.labeling import PDFLabeler
import sys

def main():
    # Procesar PDF
    if len(sys.argv) < 2:
        print("Uso: python main.py <nombre_del_pdf>")
        return

    pdf_filename = sys.argv[1]
    processor = PDFProcessor()
    pdf_info = processor.process_pdf(pdf_filename)
    
    if pdf_info:
        processor.save_pdf_info(pdf_info)
        
        # Iniciar herramienta de etiquetado
        labeler = PDFLabeler()
        labeler.run()

if __name__ == "__main__":
    main()