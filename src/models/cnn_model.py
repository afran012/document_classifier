import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (224, 224, 1),
        model_path: Optional[str] = None
    ):
        self.input_shape = input_shape
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Construye el modelo CNN
        """
        model = tf.keras.Sequential([
            # Capa de entrada
            tf.keras.layers.Input(shape=self.input_shape),
            
            # Bloque convolucional 1
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Bloque convolucional 2
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Bloque convolucional 3
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Capas densas
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model

    def compile_model(self, learning_rate: float = 0.001):
        """
        Compila el modelo
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 20,
        batch_size: int = 32,
        model_save_path: str = 'data/models/document_classifier.h5'
    ):
        """
        Entrena el modelo
        """
        X_train, y_train = train_data
        X_val, y_val = validation_data

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]

        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones
        """
        return self.model.predict(images)

    def save_model(self, path: str):
        """
        Guarda el modelo
        """
        self.model.save(path)