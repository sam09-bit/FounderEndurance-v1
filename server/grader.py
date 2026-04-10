from typing import Any

def _extract_score(payload: Any) -> float:
    """Ultra-safe score extraction to survive the automated validation bot."""
    score = 0.0
    
    # 1. Check if OpenEnv passed a full Episode object (has .state)
    if hasattr(payload, 'state'):
        state = payload.state
        if isinstance(state, dict):
            score = state.get("score", 0.0)
        else:
            score = getattr(state, "score", 0.0)
            
    # 2. Check if OpenEnv passed the state dictionary directly
    elif isinstance(payload, dict):
        score = payload.get("score", 0.0)
        
    # 3. Check if OpenEnv passed the state Pydantic object directly
    else:
        score = getattr(payload, "score", 0.0)
        
    try:
        return float(score)
    except (ValueError, TypeError):
        return 0.0

def grade_easy(state: Any = None, *args, **kwargs) -> float:
    """Grades the Easy Mode task."""
    payload = state if state is not None else kwargs.get('episode')
    score = _extract_score(payload)
    # STRICTLY clamped between 0.0 and 1.0 to pass validation
    return float(max(0.0, min(1.0, score * 1.2)))

def grade_medium(state: Any = None, *args, **kwargs) -> float:
    """Grades the Medium Mode task."""
    payload = state if state is not None else kwargs.get('episode')
    score = _extract_score(payload)
    # STRICTLY clamped between 0.0 and 1.0 to pass validation
    return float(max(0.0, min(1.0, score)))

def grade_hard(state: Any = None, *args, **kwargs) -> float:
    """Grades the Hard Mode task."""
    payload = state if state is not None else kwargs.get('episode')
    score = _extract_score(payload)
    # STRICTLY clamped between 0.0 and 1.0 to pass validation
    return float(max(0.0, min(1.0, score * 0.9)))