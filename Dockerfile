# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# CPU-only torch: the default wheel bundles CUDA and is ~2 GB larger. The cache mount makes
# retries cheap on a flaky connection.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake both Hugging Face models into the image so the container never needs the internet to run.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1

COPY *.py ./
COPY Frontend ./Frontend
COPY eval/testset.json ./eval/testset.json

# Data/ is mounted at runtime; index_store/ is a volume so the index survives restarts.
RUN useradd --create-home app && mkdir -p Data index_store eval/results && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/status' % os.environ['PORT'], timeout=6)"

CMD ["python", "app.py"]
