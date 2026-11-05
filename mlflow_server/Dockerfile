# ParkWatch — self-hosted MLflow tracking server (Azure Container App).
# Tracking store in PostgreSQL, artifacts in Blob through the managed identity, basic auth.
# The pipeline clients stay on mlflow 2.22 (pinned by azureml-mlflow): the tracking REST API
# is compatible with this 3.x server.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        "mlflow[auth]==3.18.0" \
        "psycopg[binary]==3.2.9" \
        azure-storage-blob==12.19.0 \
        azure-identity==1.19.0

RUN useradd --create-home mlflow
USER mlflow
WORKDIR /home/mlflow
COPY --chown=mlflow entrypoint.py .

EXPOSE 5000
ENTRYPOINT ["python", "entrypoint.py"]
