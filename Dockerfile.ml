# Environment for dataset building, modelling and the Streamlit app.
# Kept separate from the collector image so the collector stays tiny.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt
