#!/bin/sh
# SERVICE=api (default) or dashboard. Cloud Run provides $PORT.
if [ "$SERVICE" = "dashboard" ]; then
  exec streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0 --server.headless true
else
  exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
fi
