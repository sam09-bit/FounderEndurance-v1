# Dockerfile for FounderEndurance-v1 OpenEnv Submission
FROM python:3.10-slim

# System updates and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy codebase
COPY . /app/

# Install the custom FounderEnv package
# The colab notebook showed it had a setup.py that needs to be installed in editable mode
RUN pip install --no-cache-dir -e .

# Install inference and openenv related requirements
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    numpy>=1.24.0 \
    openai \
    openenv-core \
    fastapi \
    uvicorn

# Expose port for OpenEnv / HF Space validation pings (defaults 8080 or 7860 for HF)
EXPOSE 7860

# Optional: Add an entrypoint to launch a simple API server to fulfill the /reset HTTP validation
# For a full implementation, you should create a simple server.py that exposes `/reset` and `/step`.
# For now, this just runs a placeholder or the baseline script
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
