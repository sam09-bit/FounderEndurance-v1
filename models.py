from pydantic import BaseModel, Field

class FounderAction(BaseModel):
    work_hours_idx: int = Field(..., ge=0, le=3, description="0: 4 hours, 1: 8 hours, 2: 12 hours, 3: 16 hours")
    health_idx: int = Field(..., ge=0, le=2, description="0: Normal, 1: Drink Coffee (+velocity, +toxicity), 2: Therapy/Rest (Caps work to 8h, reduces cortisol)")
    focus_idx: int = Field(..., ge=0, le=3, description="0: Product Development, 1: Fundraising, 2: Team Building, 3: Crisis/Burnout Management")

class FounderObservation(BaseModel):
    done: bool
    reward: float
    sleep_debt: float = Field(..., description="0.0 to 1.0 (1.0 is game over)")
    cortisol_level: float = Field(..., description="0.0 to 1.0 (Stress level, 1.0 is game over)")
    caffeine_toxicity: float = Field(..., description="0.0 to 1.0")
    product_velocity: float = Field(..., description="Current speed of product development")
    team_morale: float = Field(..., description="0.0 to 1.0 (0.0 is game over)")
    cash_runway: float = Field(..., description="0.0 to 1.0 (0.0 is game over)")
    market_condition: float = Field(..., description="Fluctuating market multiplier")
    active_crisis: float = Field(..., description="1.0 if a crisis is occurring, 0.0 otherwise")
    day_of_week: float = Field(..., description="Normalized 0.0 to 1.0 (Monday to Sunday)")
    days_to_launch: float = Field(..., description="Normalized countdown to day 90")

class FounderState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    difficulty: str = "medium"
    score: float = 0.0