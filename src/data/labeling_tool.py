import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLabeler:
    def __init__(self, processed_dir='data/processed', labels_file='data/labels.json'):
        self.processed_dir = processed_dir
        self.labels_file = labels_file
        self.window = tk.Tk()
        self.window.title("PDF Page Labeler")
        self.setup_gui()
        self.load_existing_labels()

    def setup_gui(self):
        """Configura la interfaz gráfica"""
        # Frame principal
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Área de visualización
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=0, column=0, columnspan=3, pady=5)

        # Controles
        ttk.Label(self.main_frame, text="PDF actual:").grid(
            row=1, column=0, sticky=tk.W)
        self.pdf_label = ttk.Label(self.main_frame, text="")
        self.pdf_label.grid(row=1, column=1, columnspan=2, sticky=tk.W)

        ttk.Label(self.main_frame, text="Página:").grid(
            row=2, column=0, sticky=tk.W)
        self.page_label = ttk.Label(self.main_frame, text="")
        self.page_label.grid(row=2, column=1, columnspan=2, sticky=tk.W)

        # Botones
        ttk.Button(self.main_frame, text="Es Primera Página",
                   command=self.mark_first_page).grid(row=3, column=0, pady=5)
        ttk.Button(self.main_frame, text="No es Primera Página",
                   command=self.next_page).grid(row=3, column=1, pady=5)
        ttk.Button(self.main_frame, text="Guardar y Salir",
                   command=self.save_and_exit).grid(row=3, column=2, pady=5)
        # Agregar botones de navegación
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.grid(row=4, column=0, columnspan=3, pady=5)

        ttk.Button(nav_frame, text="← Anterior",
                   command=self.previous_page).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="Siguiente →",
                   command=self.next_page).grid(row=0, column=1, padx=5)

        # Mostrar estado de etiquetado
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Agregar atajos de teclado
        self.window.bind('<Left>', lambda e: self.previous_page())
        self.window.bind('<Right>', lambda e: self.next_page())
        self.window.bind('<space>', lambda e: self.mark_first_page())
        self.window.bind('q', lambda e: self.save_and_exit())

    # Nueva función para página anterior
    def previous_page(self):
        """Retrocede a la página anterior"""
        if self.current_page > 0:
            self.current_page -= 1
            self.show_current_page()

    # Mejorar la función show_current_page
    def show_current_page(self):
        """Muestra la página actual"""
        try:
            page_path = os.path.join(
                self.processed_dir,
                self.current_pdf,
                f'page_{self.current_page}.png'
            )

            if not os.path.exists(page_path):
                raise FileNotFoundError(
                    f"No se encuentra la página: {page_path}")

            # Cargar y mostrar imagen
            image = Image.open(page_path)

            # Mantener proporción de aspecto
            width, height = image.size
            ratio = min(800/width, 600/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Actualizar etiquetas
            self.pdf_label.configure(text=self.current_pdf)
            self.page_label.configure(
                text=f"Página {self.current_page + 1} de {self.get_total_pages()}")

            # Actualizar estado
            is_marked = self.is_page_marked(self.current_page)
            status = "MARCADA COMO PRIMERA PÁGINA" if is_marked else "NO MARCADA"
            self.status_label.configure(text=f"Estado: {status}")

        except Exception as e:
            logger.error(f"Error mostrando página: {e}")
            self.status_label.configure(text="Error mostrando la página")

    def get_total_pages(self):
        """Obtiene el total de páginas del PDF actual"""
        pdf_dir = os.path.join(self.processed_dir, self.current_pdf)
        return len([f for f in os.listdir(pdf_dir) if f.endswith('.png')])

    def is_page_marked(self, page_num):
        """Verifica si la página está marcada como primera página"""
        return (self.current_pdf in self.labels and
                page_num in self.labels[self.current_pdf]["target_pages"])

    def load_existing_labels(self):
        """Carga etiquetas existentes si las hay"""
        try:
            if os.path.exists(self.labels_file):
                with open(self.labels_file, 'r') as f:
                    self.labels = json.load(f)
            else:
                self.labels = {}
        except Exception as e:
            logger.error(f"Error cargando etiquetas: {e}")
            self.labels = {}

    def start_labeling(self):
        """Inicia el proceso de etiquetado"""
        pdfs = [d for d in os.listdir(self.processed_dir)
                if os.path.isdir(os.path.join(self.processed_dir, d))]

        if not pdfs:
            logger.error("No se encontraron PDFs procesados")
            return

        self.current_pdf = pdfs[0]
        self.current_page = 0
        self.show_current_page()
        self.window.mainloop()

    def show_current_page(self):
        """Muestra la página actual"""
        page_path = os.path.join(
            self.processed_dir,
            self.current_pdf,
            f'page_{self.current_page}.png'
        )

        if os.path.exists(page_path):
            # Cargar y mostrar imagen
            image = Image.open(page_path)
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Mantener referencia

            # Actualizar etiquetas
            self.pdf_label.configure(text=self.current_pdf)
            self.page_label.configure(text=f"{self.current_page + 1}")
        else:
            logger.error(f"No se encuentra la página: {page_path}")

    def mark_first_page(self):
        """Marca la página actual como primera página"""
        if self.current_pdf not in self.labels:
            self.labels[self.current_pdf] = {
                "target_pages": [], "total_pages": 0}

        if self.current_page not in self.labels[self.current_pdf]["target_pages"]:
            self.labels[self.current_pdf]["target_pages"].append(
                self.current_page)
            logger.info(
                f"Página {self.current_page} marcada como primera página")

        self.next_page()

    def next_page(self):
        """Pasa a la siguiente página"""
        self.current_page += 1
        pages = os.listdir(os.path.join(self.processed_dir, self.current_pdf))

        if self.current_page >= len(pages):
            self.labels[self.current_pdf]["total_pages"] = len(pages)
            self.save_labels()
            self.load_next_pdf()
        else:
            self.show_current_page()

    def load_next_pdf(self):
        """Carga el siguiente PDF no etiquetado"""
        pdfs = os.listdir(self.processed_dir)
        current_index = pdfs.index(self.current_pdf)

        if current_index + 1 < len(pdfs):
            self.current_pdf = pdfs[current_index + 1]
            self.current_page = 0
            self.show_current_page()
        else:
            logger.info("Todos los PDFs han sido etiquetados")
            self.save_and_exit()

    def save_labels(self):
        """Guarda las etiquetas en el archivo"""
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=4)
            logger.info("Etiquetas guardadas correctamente")
        except Exception as e:
            logger.error(f"Error guardando etiquetas: {e}")

    def save_and_exit(self):
        """Guarda las etiquetas y cierra la aplicación"""
        self.save_labels()
        self.window.quit()


if __name__ == "__main__":
    labeler = PDFLabeler()
    labeler.start_labeling()
