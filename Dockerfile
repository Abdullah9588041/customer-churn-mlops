# Use the official slim Python image; everything else is pinned in requirements.txt
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for lightgbm / shap wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# CHURN_MODEL_PATH defaults to models/champion.pkl inside the image.
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
