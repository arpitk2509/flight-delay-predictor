# Small image that serves the API + web page. Training happens on your own machine, not here.
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only code + trained artifacts go into the image (see .dockerignore: data/ is excluded)
COPY src ./src
COPY api ./api
COPY web ./web
COPY artifacts ./artifacts

# Hugging Face Spaces expects 7860, Cloud Run injects $PORT. This works for both.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
