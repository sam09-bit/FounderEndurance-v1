from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class FounderAction(Action):
    # Added defaults so validator testing empty payloads doesn't trigger a 422 crash
    work_hours_idx: int = 1
    focus_idx: int = 0
    health_idx: int = 0

class FounderObservation(Observation):
    sleep_debt: float
    cortisol_level: float
    caffeine_toxicity: float
    product_velocity: float
    team_morale: float
    cash_runway: float
    market_condition: float
    active_crisis: float
    day_of_week: float
    days_to_launch: float

class FounderState(State):
    difficulty: str = "medium"
    score: float = 0.0