"""Simulated IoT sensor feed. Run as a Cloud Run Job + Cloud Scheduler. Writes to BigQuery if BQ_SENSOR_TABLE set, else prints."""
import os, random, datetime, json
rows = [{"ts": datetime.datetime.utcnow().isoformat(), "field_id": f"field-{i}",
         "soil_moisture": round(random.uniform(15, 40), 1), "temperature": round(random.uniform(20, 35), 1),
         "humidity": round(random.uniform(45, 85), 1), "ph": round(random.uniform(5.8, 7.4), 2)} for i in range(1, 6)]
t = os.getenv("BQ_SENSOR_TABLE")
if t:
    from google.cloud import bigquery
    print(bigquery.Client().insert_rows_json(t, rows) or "inserted")
else:
    print(json.dumps(rows, indent=1))
