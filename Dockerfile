FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; chromadb needs build tooling for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# Shell form so $PORT (set by Render/Railway/Fly) is honored; falls back to 8080.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
