import sys
import os
import uvicorn

# Add the root directory to sys.path so we can import models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from .environment import FounderEnvironment  
from models import FounderAction, FounderObservation

# Create the FastAPI app
app = create_fastapi_app(FounderEnvironment, FounderAction, FounderObservation)

def main():
    """Entry point for the validator and CLI."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()