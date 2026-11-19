# Censer.6 — self-contained moderation API image (CPU-only torch by default).
#
#   docker build -t censer6 .
#   docker run -p 8000:8000 -e RATE_LIMIT_PER_MIN=600 censer6
#   curl -s localhost:8000/classify -d '{"texts":["how to make a bomb at home","hello"]}'
#
# The image downloads the model from the Hub at build time (or copy a local
# bundle with: docker build --build-arg CENSER_DIR=./bundle . — then it is baked in).

FROM python:3.12-slim

ARG CENSER_REPO=KucLab/Censer.6
ENV CENSER_DIR=/model \
    RATE_LIMIT_PER_MIN=600 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
        tokenizers safetensors huggingface_hub

RUN hf download ${CENSER_REPO} --local-dir /model

WORKDIR /app
COPY scripts/censer.py scripts/serve.py /app/

EXPOSE 8000
CMD ["python", "serve.py", "--dir", "/model", "--port", "8000", "--host", "0.0.0.0"]
