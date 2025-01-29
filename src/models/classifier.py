import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, input_shape=(224, 224, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """
        Construye el modelo CNN
        """
        model = tf.keras.Sequential([
            # Capa de entrada
            tf.keras.layers.Input(shape=self.input_shape),
            
            # Primer bloque convolucional
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Segundo bloque convolucional
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Tercer bloque convolucional
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

    def compile_model(self, learning_rate=0.001):
        """
        Compila el modelo
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

    def train(self, X_train, y_train, validation_data=None, epochs=20, batch_size=32):
        """
        Entrena el modelo
        """
        # Crear directorio para checkpoints
        checkpoint_dir = "data/models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "model_{epoch:02d}.h5"),
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        ]

        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return self.history

    def predict(self, X):
        """
        Realiza predicciones
        """
        return self.model.predict(X)

    def save_model(self, path):
        """
        Guarda el modelo
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Carga un modelo guardado
        """
        self.model = tf.keras.models.load_model(path)