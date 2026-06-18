FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user \
    PATH=/opt/conda/bin:/home/user/.local/bin:$PATH \
    SONA_SPEECH_CONFIG=speech.hf-full \
    SONA_HF_CACHE=.models \
    HF_HOME=.models \
    HUGGINGFACE_HUB_CACHE=.models/hub \
    TRANSFORMERS_CACHE=.models/transformers \
    CONDA_ALWAYS_YES=true

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user:user backend/requirements-hf.txt backend/requirements-hf.txt
COPY --chown=user:user backend/requirements-aligner.txt backend/requirements-aligner.txt
COPY --chown=user:user backend/requirements-diarization.txt backend/requirements-diarization.txt

RUN conda create -n sona-ai python=3.12 && \
    conda run -n sona-ai python -m pip install --upgrade pip setuptools wheel && \
    conda run -n sona-ai python -m pip install -r backend/requirements-hf.txt && \
    conda create -n sona-aligner python=3.12 && \
    conda run -n sona-aligner python -m pip install --upgrade pip setuptools wheel && \
    conda run -n sona-aligner python -m pip install -r backend/requirements-aligner.txt && \
    conda create -n sona-diarization python=3.12 && \
    conda run -n sona-diarization python -m pip install --upgrade pip setuptools wheel && \
    conda run -n sona-diarization python -m pip install -r backend/requirements-diarization.txt && \
    conda clean -afy

COPY --chown=user:user backend backend
COPY --chown=user:user configs configs
COPY --chown=user:user tools tools

RUN conda run -n sona-ai python -m pip install -e backend && \
    mkdir -p backend/data data/projects .models && \
    chown -R user:user /home/user/app

USER user

EXPOSE 7860

CMD ["conda", "run", "--no-capture-output", "-n", "sona-ai", "uvicorn", "sona_ai.api.main:app", "--app-dir", "backend/src", "--host", "0.0.0.0", "--port", "7860"]
