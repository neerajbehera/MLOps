import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Build Spark session with Delta Lake and Hadoop AWS support
builder = (SparkSession.builder
    .appName("DeltaMLflowExample")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ROOT_USER", "mlflow"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_ROOT_PASSWORD", "mlflow123"))
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")   # <-- ADD THIS
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("DeltaLakeExperiment")

# Create sample data and write to Delta table in MinIO
data = spark.range(0, 100)
data.write.format("delta").mode("overwrite").save("s3a://deltalake/my_table")

with mlflow.start_run() as run:
    # Get Delta table version
    delta_version = spark.sql("DESCRIBE HISTORY delta.`s3a://deltalake/my_table`").first()["version"]
    mlflow.log_param("delta_version", delta_version)
    mlflow.log_metric("row_count", data.count())

    # Prepare features for linear regression
    df = spark.range(0, 100).selectExpr("id as label", "cast(id as double) as feature")
    assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
    df_vector = assembler.transform(df).select("label", "features")

    # Train model
    lr = LinearRegression(maxIter=10, labelCol="label", featuresCol="features")
    model = lr.fit(df_vector)

    # Log model
    mlflow.spark.log_model(model, "model")

    # Save delta version as artifact
    with open("/tmp/delta_version.txt", "w") as f:
        f.write(str(delta_version))
    mlflow.log_artifact("/tmp/delta_version.txt")

    print(f"Run ID: {run.info.run_id}")

spark.stop()