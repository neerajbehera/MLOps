import os
import gc
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ------------------------------------------------------------
# 1. Spark Session with Delta Lake and S3A (MinIO) support
# ------------------------------------------------------------
builder = (SparkSession.builder
    .appName("IrisClassifier")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ROOT_USER"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_ROOT_PASSWORD"))
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")   # for MLflow s3://
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# ------------------------------------------------------------
# 2. Read Iris CSV from MinIO (assumes no header, all numeric)
# ------------------------------------------------------------
df = spark.read.option("header", "false").csv("s3a://data/iris_data.csv")
df = df.toDF("sepal_length", "sepal_width", "petal_length", "petal_width", "species") \
       .withColumn("sepal_length", col("sepal_length").cast(DoubleType())) \
       .withColumn("sepal_width", col("sepal_width").cast(DoubleType())) \
       .withColumn("petal_length", col("petal_length").cast(DoubleType())) \
       .withColumn("petal_width", col("petal_width").cast(DoubleType()))

# ------------------------------------------------------------
# 3. Write raw data to Delta table (versioned)
# ------------------------------------------------------------
delta_path = "s3a://deltalake/iris"
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(delta_path)

# ------------------------------------------------------------
# 4. Feature preparation
# ------------------------------------------------------------
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                            outputCol="features")
indexer = StringIndexer(inputCol="species", outputCol="label")
df_ready = assembler.transform(df)
df_ready = indexer.fit(df_ready).transform(df_ready)

# ------------------------------------------------------------
# 5. Train / test split
# ------------------------------------------------------------
train, test = df_ready.randomSplit([0.8, 0.2], seed=42)

# ------------------------------------------------------------
# 6. Train Random Forest
# ------------------------------------------------------------
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)
model = rf.fit(train)

# ------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------
predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# ------------------------------------------------------------
# 8. MLflow Tracking
# ------------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("IrisExperiment")

with mlflow.start_run() as run:
    # Delta version
    delta_version = spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`").first()["version"]
    mlflow.log_param("delta_version", delta_version)
    mlflow.log_param("num_trees", 50)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.spark.log_model(model, "random-forest-model")

    # Log delta version as artifact
    with open("/tmp/delta_version.txt", "w") as f:
        f.write(str(delta_version))
    mlflow.log_artifact("/tmp/delta_version.txt")

    print(f"Run ID: {run.info.run_id}")
    print(f"MLflow UI: http://localhost:5001/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

# ------------------------------------------------------------
# 9. Graceful cleanup (suppress harmless __del__ exceptions)
# ------------------------------------------------------------
# Delete explicit references to Spark objects
del model
del assembler, indexer, df, df_ready, train, test, predictions

# Force garbage collection
gc.collect()

# Stop Spark – any remaining __del__ calls will now be harmless
spark.stop()