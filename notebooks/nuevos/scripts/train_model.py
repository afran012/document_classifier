import os
import sys
import json
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as F

# Configuración del entorno
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Configuración de rutas
PROCESSED_DIR = 'data/processed'
LABELS_JSON = 'data/labels.json'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Inicializar Spark
spark = SparkSession.builder \
    .appName("DocumentClassifier") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        resized = cv2.resize(image, target_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized.flatten().tolist()
    except Exception as e:
        print(f"Error procesando {image_path}: {str(e)}")
        return None

# Cargar y procesar datos
with open(LABELS_JSON, 'r') as f:
    labels = json.load(f)

image_data = []
for pdf_name, pdf_info in labels.items():
    for page_num in range(pdf_info["total_pages"]):
        image_path = os.path.join(PROCESSED_DIR, pdf_name, f'page_{page_num}.png')
        if os.path.exists(image_path):
            label = 1 if page_num in pdf_info["target_pages"] else 0
            features = preprocess_image(image_path)
            if features is not None:
                image_data.append((image_path, label, features))

# Crear DataFrame
schema = StructType([
    StructField("path", StringType(), False),
    StructField("label", IntegerType(), False),
    StructField("features", ArrayType(FloatType()), True)
])

df = spark.createDataFrame(image_data, schema)

# Convertir features a vector
array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
df = df.withColumn("features_vector", array_to_vector_udf("features"))

# Escalar características
scaler = StandardScaler(inputCol="features_vector", 
                       outputCol="scaled_features",
                       withStd=True,
                       withMean=True)
scaler_model = scaler.fit(df)
df_scaled = scaler_model.transform(df)

# Sobremuestreo de la clase minoritaria
minority_class_df = df_scaled.filter(col("label") == 1)
majority_class_df = df_scaled.filter(col("label") == 0)

# Número de ejemplos en la clase mayoritaria
majority_count = majority_class_df.count()

# Sobremuestrear la clase minoritaria
oversampled_minority_class_df = minority_class_df.sample(withReplacement=True, fraction=majority_count / minority_class_df.count())

# Combinar los DataFrames
balanced_df = majority_class_df.union(oversampled_minority_class_df)

# Verificar la nueva distribución de clases
balanced_df.groupBy("label").count().show()

# Preparar datos para entrenamiento
final_df = balanced_df.select("label", "scaled_features")
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# Verificar datos antes de entrenar
print(f"Total de registros en el conjunto de entrenamiento: {train_df.count()}")
print(f"Total de registros en el conjunto de prueba: {test_df.count()}")

# Entrenar modelo
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="label",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0.8
)

try:
    lr_model = lr.fit(train_df)
except Exception as e:
    print(f"Error durante el entrenamiento del modelo: {str(e)}")
    spark.stop()
    sys.exit(1)

# Evaluar modelo
predictions = lr_model.transform(test_df)

# Evaluar con métricas adicionales
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

accuracy = evaluator_accuracy.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"\nMétricas del modelo:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Guardar modelo y datos procesados
try:
    lr_model.write().overwrite().save(os.path.join(MODEL_DIR, "logistic_regression_model"))
    scaler_model.write().overwrite().save(os.path.join(PROCESSED_DIR, "scaler_model"))
except Exception as e:
    print(f"Error al guardar el modelo: {str(e)}")

try:
    df_scaled.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "processed_scaled_data.parquet"))
except Exception as e:
    print(f"Error al guardar los datos procesados: {str(e)}")

# Mostrar estadísticas
print("\nEstadísticas del procesamiento:")
print(f"Total imágenes: {df.count()}")
print("\nDistribución de clases:")
df.groupBy("label").count().show()

spark.stop()