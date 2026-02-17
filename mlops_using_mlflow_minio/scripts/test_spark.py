import os
from pyspark.sql import SparkSession

print("JAVA_HOME:", os.environ.get("JAVA_HOME", "Not set"))
print("SPARK_HOME:", os.environ.get("SPARK_HOME", "Not set"))

try:
    spark = SparkSession.builder.appName("MinimalTest").master("local[1]").getOrCreate()
    print("✓ Spark session created successfully")
    print("Spark version:", spark.version)
    spark.stop()
except Exception as e:
    print("✗ Failed to create Spark session:")
    print(e)