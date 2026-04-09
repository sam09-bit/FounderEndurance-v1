def grade_easy_mode(state_score: float) -> float:
    """Grader for the Easy Mode task."""
    # In easy mode, we are generous with the score
    return min(1.0, state_score * 1.2)

def grade_medium_mode(state_score: float) -> float:
    """Grader for the Medium Mode task."""
    # In medium mode, the score is exactly as calculated
    return min(1.0, max(0.0, state_score))

def grade_hard_mode(state_score: float) -> float:
    """Grader for the Hard Mode task."""
    # In hard mode, it's very difficult to get a perfect score
    # We apply a slight penalty curve
    return min(1.0, max(0.0, state_score * 0.9))

def get_score_for_task(difficulty: str, raw_score: float) -> float:
    """Routes the raw score to the appropriate task grader."""
    if difficulty == "easy":
        return grade_easy_mode(raw_score)
    elif difficulty == "hard":
        return grade_hard_mode(raw_score)
    else:
        return grade_medium_mode(raw_score)