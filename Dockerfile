FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the leak-safe DuckDB/Parquet feature store from the canonical CSV data
# during the image build. The artifacts are intentionally gitignored, but are
# present in the deployed Cloud Run image and revalidated at model startup.
RUN python build_bigdata.py

EXPOSE 8080

CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
