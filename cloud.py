"""Optional Google Cloud helpers. No-ops unless env vars are set, so local runs need no credentials.
MODEL_BUCKET=my-bucket        -> missing files in models/ are fetched from gs://my-bucket/models/
BQ_TABLE=project.dataset.tbl  -> predictions are logged to BigQuery"""
import datetime, json, os
from pathlib import Path

def ensure_model(path):
    path = Path(path); bucket = os.getenv("MODEL_BUCKET")
    if path.exists() or not bucket:
        return
    try:
        from google.cloud import storage
        path.parent.mkdir(parents=True, exist_ok=True)
        storage.Client().bucket(bucket).blob("models/" + path.name).download_to_filename(str(path))
    except Exception as e:
        print("Model download failed:", path.name, e)

def log_prediction(kind, inputs, output):
    table = os.getenv("BQ_TABLE")
    if not table:
        return
    try:
        from google.cloud import bigquery
        bigquery.Client().insert_rows_json(table, [{"ts": datetime.datetime.utcnow().isoformat(),
            "kind": kind, "inputs": json.dumps(inputs, default=str), "output": json.dumps(output, default=str)}])
    except Exception as e:
        print("BigQuery log failed:", e)
