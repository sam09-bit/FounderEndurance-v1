FROM python:3.10-slim

WORKDIR /app
COPY . /app/

RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    openai \
    openenv-core \
    fastapi \
    uvicorn \
    pydantic

# Dynamic port binding handles both HF Spaces and the automated validation checker
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]