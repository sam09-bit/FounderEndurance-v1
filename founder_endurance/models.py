from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class FounderAction(Action):
    work_hours_idx: int
    focus_idx: int
    health_idx: int

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