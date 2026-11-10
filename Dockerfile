FROM python:3.11-slim
WORKDIR /srv
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY *.py start.sh ./
# models are fetched from Cloud Storage at start-up (MODEL_BUCKET), not baked into the image
CMD ["sh", "start.sh"]
