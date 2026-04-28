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

# A single image runs both the FastAPI service and the Streamlit demo. Pick
# which one with the ``RUN_MODE`` env variable (``api`` -- default -- or
# ``ui``). ``PORT`` is read at runtime so the same image works on Render,
# Railway, Fly, etc.
ENV RUN_MODE=api \
    PORT=8000

EXPOSE 8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD sh -c 'if [ "$RUN_MODE" = "ui" ]; then \
      curl -fsS http://localhost:${PORT:-8501}/_stcore/health || exit 1; \
    else \
      curl -fsS http://localhost:${PORT:-8000}/health || exit 1; \
    fi'

CMD ["sh", "-c", "if [ \"$RUN_MODE\" = \"ui\" ]; then \
    exec streamlit run scripts/product_demo.py \
      --server.port=${PORT:-8501} \
      --server.address=0.0.0.0 \
      --server.headless=true \
      --browser.gatherUsageStats=false; \
  else \
    exec uvicorn scripts.production_api:app \
      --host 0.0.0.0 --port ${PORT:-8000}; \
  fi"]
