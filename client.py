from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import FounderAction, FounderObservation, FounderState  # ✅ absolute import

class FounderEnvClient(EnvClient[FounderAction, FounderObservation, FounderState]):
    def _step_payload(self, action: FounderAction) -> dict:
        return {
            "work_hours_idx": action.work_hours_idx,
            "focus_idx": action.focus_idx,
            "health_idx": action.health_idx
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=FounderObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                sleep_debt=obs_data.get("sleep_debt", 0.0),
                cortisol_level=obs_data.get("cortisol_level", 0.0),
                caffeine_toxicity=obs_data.get("caffeine_toxicity", 0.0),
                product_velocity=obs_data.get("product_velocity", 0.0),
                team_morale=obs_data.get("team_morale", 0.0),
                cash_runway=obs_data.get("cash_runway", 0.0),
                market_condition=obs_data.get("market_condition", 0.0),
                active_crisis=obs_data.get("active_crisis", 0.0),
                day_of_week=obs_data.get("day_of_week", 0.0),
                days_to_launch=obs_data.get("days_to_launch", 0.0),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> FounderState:
        return FounderState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "medium"),
            score=payload.get("score", 0.0)
        )