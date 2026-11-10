#!/usr/bin/env bash
# Usage (Git Bash/WSL/Linux/Mac, from project root, after `gcloud auth login` and `python train_and_save.py`):
#   PROJECT=my-project REGION=asia-south1 ./deploy/deploy.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"; REGION=${REGION:-asia-south1}; BUCKET=${PROJECT}-agrisense
[ -f models/meta.json ] || { echo "models/ is empty - run python train_and_save.py first"; exit 1; }
gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com bigquery.googleapis.com storage.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gsutil mb -l "$REGION" "gs://$BUCKET" 2>/dev/null || true
gsutil -m cp models/* "gs://$BUCKET/models/"
bq mk -d agrisense 2>/dev/null || true
bq mk -t agrisense.predictions ts:TIMESTAMP,kind:STRING,inputs:STRING,output:STRING 2>/dev/null || true
bq mk -t agrisense.sensors ts:TIMESTAMP,field_id:STRING,soil_moisture:FLOAT,temperature:FLOAT,humidity:FLOAT,ph:FLOAT 2>/dev/null || true
ENVV="MODEL_BUCKET=$BUCKET,BQ_TABLE=$PROJECT.agrisense.predictions,GEE_PROJECT=$PROJECT"
gcloud run deploy agrisense-api --source . --region "$REGION" --allow-unauthenticated --memory 2Gi --cpu 2 \
  --set-env-vars "$ENVV,SERVICE=api"
gcloud run deploy agrisense-dashboard --source . --region "$REGION" --allow-unauthenticated --memory 2Gi --cpu 2 \
  --set-env-vars "$ENVV,SERVICE=dashboard"
echo "Done. URLs:"; gcloud run services list --region "$REGION" --format 'value(metadata.name,status.url)'
echo "If the services can't read the bucket or write BigQuery, grant their service account Storage Object Viewer and BigQuery Data Editor."
