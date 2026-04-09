from openenv.core.env_server import create_fastapi_app
from server.environment import FounderEnvironment
from founder_endurance.models import FounderAction, FounderObservation

app = create_fastapi_app(FounderEnvironment, FounderAction, FounderObservation)