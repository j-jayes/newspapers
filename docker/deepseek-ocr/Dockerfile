FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl git && \
    rm -rf /var/lib/apt/lists/*

# Python deps — DeepSeek-OCR-2 requires transformers 4.x + flash-attn
RUN pip3 install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir \
    "transformers==4.46.3" \
    "tokenizers==0.20.3" \
    "accelerate>=1.0" \
    "pillow>=10.0" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.34" \
    "pydantic>=2.0" \
    "torchvision==0.21.0" \
    "addict" \
    "easydict" \
    "einops" \
    "matplotlib" \
    "requests"

# flash-attn must be compiled (requires CUDA devel headers)
RUN pip3 install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

COPY server.py /app/server.py

WORKDIR /app
EXPOSE 8000

# HF Inference Endpoints mounts model at /repository
ENV MODEL_PATH=/repository
ENV MAX_NEW_TOKENS=4096

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "server.py"]
