import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# ---------- Spark Session (same as before) ----------
builder = (SparkSession.builder
    .appName("IrisTuning")
    .master("spark://spark-master:7077")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ROOT_USER"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_ROOT_PASSWORD"))
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# ---------- Load and Prepare Data ----------
df = spark.read.option("header", "false").csv("s3a://data/iris_data.csv")
df = df.toDF("sepal_length", "sepal_width", "petal_length", "petal_width", "species") \
       .withColumn("sepal_length", col("sepal_length").cast(DoubleType())) \
       .withColumn("sepal_width", col("sepal_width").cast(DoubleType())) \
       .withColumn("petal_length", col("petal_length").cast(DoubleType())) \
       .withColumn("petal_width", col("petal_width").cast(DoubleType()))

# Write to Delta (optional, for versioning)
delta_path = "s3a://deltalake/iris"
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(delta_path)

# Feature preparation (reused for all runs)
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                            outputCol="features")
indexer = StringIndexer(inputCol="species", outputCol="label")
df_features = assembler.transform(df)
df_features = indexer.fit(df_features).transform(df_features)

# Train/validation split (same seed for fair comparison)
train, test = df_features.randomSplit([0.8, 0.2], seed=42)

# ---------- MLflow Setup ----------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("IrisTuning")

# Hyperparameters to try
num_trees_values = [10, 50, 100]

for nt in num_trees_values:
    with mlflow.start_run(run_name=f"rf_trees_{nt}") as run:
        # Train model with current hyperparameter
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=nt)
        model = rf.fit(train)
        
        # Evaluate
        predictions = model.transform(test)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        # Log parameters, metrics, and model
        mlflow.log_param("num_trees", nt)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.spark.log_model(model, f"random-forest-model-{nt}")
        
        # Also log delta version (same for all runs)
        delta_version = spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`").first()["version"]
        mlflow.log_param("delta_version", delta_version)
        
        print(f"Run {run.info.run_id}: numTrees={nt}, accuracy={accuracy:.4f}")

spark.stop()