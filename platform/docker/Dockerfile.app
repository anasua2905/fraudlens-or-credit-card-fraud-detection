# FraudLens deployable application image: the trained champion model (dashboard/bundle),
# the FastAPI scoring API and the Streamlit dashboard in one image.
#
#   docker run -p 8000:8000 ghcr.io/<owner>/fraudlens-app api         # scoring API  (/docs)
#   docker run -p 8501:8501 ghcr.io/<owner>/fraudlens-app dashboard   # dashboard
#
# Library versions come from requirements.lock.txt (the versions that trained the model).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 fraudlens
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1 \
    FRAUDLENS_MODEL_DIR=/app/dashboard/bundle FRAUDLENS_FEEDBACK_FILE=/tmp/feedback/feedback.csv
COPY pyproject.toml README.md requirements.lock.txt ./
COPY src ./src
RUN pip install -c requirements.lock.txt -e ".[ml,serve]"
COPY config ./config
COPY dashboard ./dashboard
COPY platform/docker/entrypoint.sh /usr/local/bin/fraudlens
RUN chmod +x /usr/local/bin/fraudlens && mkdir -p /tmp/feedback && chown -R fraudlens /app /tmp/feedback
USER fraudlens
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -fsS http://localhost:8000/health || curl -fsS http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["fraudlens"]
CMD ["api"]
