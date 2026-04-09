import sys
import os
# Add the root directory to sys.path so we can import models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from .environment import FounderEnvironment  
from models import FounderAction, FounderObservation

app = create_fastapi_app(FounderEnvironment, FounderAction, FounderObservation)