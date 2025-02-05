import os
import sys
import ctypes
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StandardScalerModel

def check_hadoop_libraries():
    hadoop_home = os.environ.get('HADOOP_HOME')
    if not hadoop_home:
        print("Error: HADOOP_HOME no está configurado.")
        return False

    # Verificar la existencia de winutils.exe
    winutils_path = os.path.join(hadoop_home, 'bin', 'winutils.exe')
    if not os.path.exists(winutils_path):
        print(f"Error: {winutils_path} no existe.")
        return False

    # Verificar la existencia de hadoop.dll
    hadoop_dll_path = os.path.join(hadoop_home, 'bin', 'hadoop.dll')
    if not os.path.exists(hadoop_dll_path):
        print(f"Error: {hadoop_dll_path} no existe.")
        return False

    # Intentar cargar hadoop.dll
    try:
        ctypes.cdll.LoadLibrary(hadoop_dll_path)
        print("Biblioteca nativa de Hadoop cargada correctamente.")
    except OSError as e:
        print(f"Error: No se pudo cargar la biblioteca nativa de Hadoop: {e}")
        return False

    return True

# Verificar las bibliotecas necesarias de Hadoop antes de ejecutar el script
if not check_hadoop_libraries():
    sys.exit(1)

# Configuración del entorno
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Configurar las bibliotecas nativas de Hadoop
os.environ['HADOOP_HOME'] = 'C:\\Program Files\\winutils'
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-21'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['HADOOP_HOME'], 'bin')
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['JAVA_HOME'], 'bin')

# Inicializar Spark
spark = SparkSession.builder \
    .appName("PDFFirstPagePredictor") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.executor.extraJavaOptions", "-Djava.library.path=" + os.path.join(os.environ['HADOOP_HOME'], 'bin')) \
    .getOrCreate()

def predict_first_pages(pdf_dir, model_path, scaler_path):
    # Verificar si el modelo y el escalador existen
    if not os.path.exists(model_path):
        print(f"Error: El modelo no existe en la ruta {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"Error: El escalador no existe en la ruta {scaler_path}")
        return

    # Cargar el modelo y el escalador
    lr_model = LogisticRegressionModel.load(model_path)
    scaler_model = StandardScalerModel.load(scaler_path)

    # Procesar las páginas del PDF
    image_data = []
    for page_num in range(len(os.listdir(pdf_dir))):
        image_path = os.path.join(pdf_dir, f'page_{page_num}.png')
        if os.path.exists(image_path):
            # Leer la imagen y convertirla a un vector
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized = cv2.resize(image, (224, 224))
                normalized = resized.astype(np.float32) / 255.0
                features = normalized.flatten().tolist()
                image_data.append((image_path, features))

    # Crear DataFrame
    schema = StructType([
        StructField("path", StringType(), False),
        StructField("features", ArrayType(FloatType()), True)
    ])
    df = spark.createDataFrame(image_data, schema)

    # Convertir features a vector
    array_to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    df = df.withColumn("features_vector", array_to_vector_udf("features"))

    # Escalar características
    df_scaled = scaler_model.transform(df)

    # Realizar predicciones
    predictions = lr_model.transform(df_scaled)

    # Filtrar las primeras páginas predichas
    first_pages = predictions.filter(col("prediction") == 1).select("path").collect()

    # Mostrar resultados
    print("Páginas predichas como primeras páginas:")
    for row in first_pages:
        print(row["path"])

# Ejemplo de uso
pdf_dir = 'data/pdf_pages'
model_path = 'data/models/logistic_regression_model'
scaler_path = 'data/processed/scaler_model'
predict_first_pages(pdf_dir, model_path, scaler_path)

spark.stop()