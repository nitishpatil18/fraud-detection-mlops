# fraud detection api image
FROM python:3.11-slim

WORKDIR /app

# install system deps needed for xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps (separate layer for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ ./src/

# api config via env (overridable at `docker run` time)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    DECISION_THRESHOLD=0.5 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]