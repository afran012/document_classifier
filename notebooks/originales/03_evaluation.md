# Evaluación del Modelo

Este notebook evalúa el rendimiento del modelo entrenado y visualiza los resultados.

## 1. Configuración inicial

```python
import os
import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle

# Iniciar Spark
spark = SparkSession.builder \
    .appName("ModelEvaluation") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Configurar paths
MODEL_DIR = 'data/models'
PROCESSED_DIR = 'data/processed'
```

## 2. Cargar modelo y datos

```python
# Cargar modelo
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "final_model.h5"))

# Cargar historia de entrenamiento
with open(os.path.join(MODEL_DIR, "training_history.pkl"), "rb") as f:
    history = pickle.load(f)

# Cargar datos de validación
df_val = spark.read.parquet(os.path.join(PROCESSED_DIR, "processed_pages.parquet"))
val_data = df_val.toPandas()

# Preparar datos
X_val = np.array([x for x in val_data["features"]])
y_val = val_data["label"].values

# Realizar predicciones
y_pred = model.predict(X_val)
y_pred_classes = (y_pred > 0.5).astype(int)
```

## 3. Visualizar métricas de entrenamiento

```python
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)
```

## 4. Evaluar rendimiento del modelo

```python
# Matriz de confusión
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Etiqueta Verdadera')
plt.xlabel('Etiqueta Predicha')
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_val, y_pred_classes))

# Curva ROC
fpr, tpr, _ = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
```

## 5. Análisis de errores

```python
def analyze_errors(X_val, y_val, y_pred_classes, num_examples=5):
    errors = np.where(y_val