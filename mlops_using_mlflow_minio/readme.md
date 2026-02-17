# Start all containers in detached mode
docker-compose up -d


# create miniIo buckets
docker exec -it mlops-minio mc alias set local http://localhost:9000 mlflow mlflow123
docker exec -it mlops-minio mc mb local/deltalake
docker exec -it mlops-minio mc mb local/mlflow

# test spark connnection
docker exec -it mlops-jupyter python /home/jovyan/scripts/test_spark.py


# run end to end test
docker exec -it mlops-jupyter python /home/jovyan/scripts/train.py


# verify results
MLflow UI: http://localhost:5001 – find experiment DeltaLakeExperiment and inspect the run.

MinIO Console: http://localhost:8900 – browse deltalake and mlflow buckets.


## For Iris data set
 Prepare the Iris Dataset
Ensure the Iris CSV file (iris_data.csv) is available on your host.

Upload it to MinIO by creating a bucket named data and placing the file there as iris_data.csv.
The file should contain the classic Iris measurements (sepal length, sepal width, petal length, petal width, species) with no header row.

2. Run the Training Script
The script train_iris.py (final version with proper imports, Delta overwrite, and graceful cleanup) must be located in the scripts/ folder, which is mounted to the Jupyter container.

Execute the script from within the Jupyter container. It will:

Read the CSV directly from MinIO (s3a://data/iris_data.csv).

Write the raw data as a Delta table to s3a://deltalake/iris (schema overwritten to ensure compatibility).

Prepare features, train a Random Forest classifier (50 trees), and evaluate accuracy.

Log all parameters (delta_version, num_trees), metrics (accuracy), the model itself, and a delta_version.txt artifact to MLflow.

Print the Run ID and MLflow UI link.

3. Verify the Results
MLflow UI
Open http://localhost:5001. Look for the experiment named IrisExperiment. Click on the most recent run to inspect:

Parameters: delta_version, num_trees

Metric: accuracy

Artifacts: random-forest-model/ (model files) and delta_version.txt

MinIO Console
Open http://localhost:8900 (login mlflow / mlflow123). Verify:

Bucket deltalake/iris/ contains Parquet data files and a _delta_log/ directory (transaction logs). This confirms the data is versioned.

Bucket mlflow/2/<run_id>/artifacts/ contains the logged model and the delta_version.txt file.


