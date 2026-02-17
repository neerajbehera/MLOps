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

