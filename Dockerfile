# GPU image for Azure Container Apps (CUDA 12.8, cuDNN 9).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-server.txt

COPY predict.py server.py ./

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
