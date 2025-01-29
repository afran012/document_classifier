import json
import os
from PIL import Image
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk

class PDFLabeler:
    def __init__(self, pdf_info_file='labels/pdf_info.json'):
        self.pdf_info = self.load_pdf_info(pdf_info_file)
        self.labels = {}
        self.current_pdf = None
        self.setup_gui()

    def load_pdf_info(self, pdf_info_file):
        with open(pdf_info_file, 'r') as f:
            return json.load(f)

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("PDF Page Labeler")
        self.window.geometry("800x600")

        # Controles
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(control_frame, text="Página actual:").pack(side='left')
        self.page_label = ttk.Label(control_frame, text="0/0")
        self.page_label.pack(side='left', padx=5)

        ttk.Button(control_frame, text="Marcar como Primera Página", 
                  command=self.mark_positive).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Marcar como Otra Página", 
                  command=self.mark_negative).pack(side='left', padx=5)

        # Área de imagen
        self.image_label = ttk.Label(self.window)
        self.image_label.pack(expand=True, fill='both')

        self.load_next_pdf()

    def load_next_pdf(self):
        # Cargar siguiente PDF no etiquetado
        for pdf_name, info in self.pdf_info.items():
            if pdf_name not in self.labels:
                self.current_pdf = pdf_name
                self.current_page = 0
                self.total_pages = info['total_pages']
                self.show_current_page()
                break

    def show_current_page(self):
        if not self.current_pdf:
            return

        page_info = self.pdf_info[self.current_pdf]['pages'][self.current_page]
        image_path = page_info['image_path']

        # Cargar y mostrar imagen
        image = Image.open(image_path)
        image.thumbnail((700, 500))
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.page_label.configure(text=f"{self.current_page + 1}/{self.total_pages}")

    def mark_positive(self):
        if not self.current_pdf:
            return

        if self.current_pdf not in self.labels:
            self.labels[self.current_pdf] = []
        
        self.labels[self.current_pdf].append(self.current_page)
        self.next_page()

    def mark_negative(self):
        self.next_page()

    def next_page(self):
        if not self.current_pdf:
            return

        self.current_page += 1
        if self.current_page >= self.total_pages:
            self.save_labels()
            self.load_next_pdf()
        else:
            self.show_current_page()

    def save_labels(self):
        output_file = os.path.join('labels', 'page_labels.json')
        with open(output_file, 'w') as f:
            json.dump(self.labels, f, indent=4)
        print(f"Etiquetas guardadas en {output_file}")

    def run(self):
        self.window.mainloop()