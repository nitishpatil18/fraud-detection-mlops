# fraud detection api image, with model baked in.
# build locally with:
#   MODEL_RUN_ID=<run_id> python -m scripts.export_model
#   docker build -t fraud-api:latest .

FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY src/ ./src/

# copy the pre-exported model artifacts.
# must run `python -m scripts.export_model` with MODEL_RUN_ID set before docker build.
COPY build/model/ ./model/

# runtime env
ENV MODEL_LOCAL_DIR=/app/model \
    DECISION_THRESHOLD=0.5 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]