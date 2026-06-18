FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    SONA_SPEECH_CONFIG=speech.hf \
    SONA_HF_CACHE=.models \
    HF_HOME=.models \
    HUGGINGFACE_HUB_CACHE=.models/hub \
    TRANSFORMERS_CACHE=.models/transformers

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user
WORKDIR /home/user/app

COPY --chown=user:user backend/requirements-hf.txt backend/requirements-hf.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r backend/requirements-hf.txt

COPY --chown=user:user backend backend
COPY --chown=user:user configs configs

RUN pip install -e backend && \
    mkdir -p backend/data data/projects .models

EXPOSE 7860

CMD ["uvicorn", "sona_ai.api.main:app", "--app-dir", "backend/src", "--host", "0.0.0.0", "--port", "7860"]
