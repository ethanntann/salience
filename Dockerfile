FROM node:22-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend ./
RUN npm run build

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SALIENCE_DEMO_MODE=true
ENV SALIENCE_APP_DATA_DIR=/tmp/salience
ENV SALIENCE_STUDENT_ARTIFACTS_DIR=/app/student-artifacts

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/salience_api /app/backend/salience_api
COPY demo-data /app/demo-data
COPY student-artifacts /app/student-artifacts
COPY sample-clips /app/sample-clips
COPY demo-video /app/demo-video
COPY --from=frontend /app/frontend/dist /app/frontend/dist

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /app/backend && python -m pip install -e .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn salience_api.app:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir /app/backend"]
