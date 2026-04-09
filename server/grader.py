from typing import Any

def _extract_score(state: Any) -> float:
    """Helper to safely extract the score whether OpenEnv passes a dict or an object."""
    try:
        # If state is passed as a Pydantic object
        return float(state.score)
    except AttributeError:
        # If state is passed as a standard dictionary
        if isinstance(state, dict):
            return float(state.get("score", 0.0))
    return 0.0

def grade_easy(state: Any, *args, **kwargs) -> float:
    """Grades the Easy Mode task."""
    score = _extract_score(state)
    return min(1.0, score * 1.2)  # Slightly generous curve for easy mode

def grade_medium(state: Any, *args, **kwargs) -> float:
    """Grades the Medium Mode task."""
    score = _extract_score(state)
    return max(0.0, min(1.0, score))  # Standard score

def grade_hard(state: Any, *args, **kwargs) -> float:
    """Grades the Hard Mode task."""
    score = _extract_score(state)
    return max(0.0, min(1.0, score * 0.9))  # Harsher curve for hard mode