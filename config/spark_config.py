from pyspark.sql import SparkSession

def create_spark_session(app_name="PDFClassifier", memory="4g"):
    """
    Crea y configura una sesión de Spark
    """
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.executor.memory", memory)
            .config("spark.driver.memory", memory)
            .config("spark.driver.maxResultSize", memory)
            .getOrCreate())

def get_spark_session():
    """
    Obtiene la sesión de Spark actual o crea una nueva
    """
    try:
        return SparkSession.builder.getOrCreate()
    except Exception as e:
        print(f"Error creando sesión Spark: {str(e)}")
        return create_spark_session()