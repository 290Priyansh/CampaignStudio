# Reproducible runtime environment for the AI Creative Director.
#
# IMPORTANT (see README "Docker" section): this image does NOT bundle
# Ollama, and it does not solve GPU access by itself. It packages the
# Python application only. You still need:
#   1. Ollama running somewhere reachable from this container
#      (typically on the host -- see docker-compose.yml).
#   2. For real (non-toy) diffusion speed, the host needs an NVIDIA GPU +
#      the NVIDIA Container Toolkit, and this container needs to be run
#      with `--gpus all` (or the docker-compose GPU block uncommented).
#      PyPI's torch wheel bundles the CUDA runtime it needs, so a plain
#      python:3.11-slim base is sufficient -- no CUDA base image required.

FROM python:3.11-slim

# libgl1: required by Pillow/opencv-adjacent codepaths some diffusers
# preprocessing pulls in. build-essential: needed for a couple of source
# builds (e.g. some open-clip-torch dependency versions).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
