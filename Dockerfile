FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch (much smaller than the default version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Download the model from the Hugging Face Hub while building the image
ARG MODEL_ID=MithunKaveriappa/banking77-distilbert
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')"
ENV MODEL_PATH=${MODEL_ID} \
    HF_HUB_OFFLINE=1

COPY src ./src
COPY app ./app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]