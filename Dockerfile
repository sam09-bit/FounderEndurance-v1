FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables for HF Spaces
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

# FIXED: Pointing uvicorn to the server/ folder!
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]