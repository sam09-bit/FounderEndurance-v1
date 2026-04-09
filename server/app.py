from openenv.core.env_server import create_fastapi_app
from .environment import FounderEnvironment  # Notice the dot here!
from models import FounderAction, FounderObservation

app = create_fastapi_app(FounderEnvironment, FounderAction, FounderObservation)