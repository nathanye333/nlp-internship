# NOTE: we use the full ``python:3.11`` image rather than ``-slim``. The
# current ``python:3.11-slim`` arm64 manifest fails with
# ``exec /bin/sh: exec format error`` under Docker Desktop on Apple Silicon;
# the full image already ships ``build-essential`` + ``curl`` so we also no
# longer need an extra ``apt-get`` step.
FROM python:3.11 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY scripts ./scripts
COPY data ./data
COPY docs ./docs
COPY tests ./tests
COPY pytest.ini ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "scripts.production_api:app", "--host", "0.0.0.0", "--port", "8000"]
