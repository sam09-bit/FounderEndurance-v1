"""
FounderEndurance-v1 Inference Script
Emits [START], [STEP], [END] logs as required by OpenEnv validation.
"""

import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import FounderEnvClient          # ✅ correct class name
from models import FounderAction

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL    = os.getenv("SPACE_URL", "http://0.0.0.0:7860")
TASK_NAME    = os.getenv("FOUNDER_TASK", "survive_medium")
BENCHMARK    = "founder-endurance-env"
MAX_STEPS    = 90
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent playing the role of a startup founder for 90 days.
    Survive by managing: cash_runway, team_morale, cortisol_level, sleep_debt, product_velocity.

    Fatal conditions (episode ends immediately):
    - cash_runway <= 0
    - team_morale <= 0
    - cortisol_level >= 1.0
    - sleep_debt >= 1.0

    Win condition: survive 90 days with product_velocity > 0.8 and cash_runway > 0.

    Action format — return ONLY valid JSON, no markdown, no explanation:
    {"work_hours_idx": <0-3>, "health_idx": <0-2>, "focus_idx": <0-3>}

    work_hours_idx: 0=4h, 1=8h, 2=12h, 3=16h
    health_idx:     0=Normal, 1=Drink Coffee, 2=Therapy/Rest
    focus_idx:      0=Product, 1=Fundraising, 2=Team Building, 3=Crisis/Burnout Mgmt

    Decision rules:
    - active_crisis=1.0 → focus_idx=3, health_idx=2, work_hours_idx=1
    - cortisol > 0.7    → focus_idx=3 or health_idx=2
    - sleep_debt > 0.6  → work_hours_idx=0, health_idx=2
    - cash_runway < 0.3 → focus_idx=1
    - team_morale < 0.3 → focus_idx=2
    - caffeine > 0.5    → health_idx=0 (no coffee)
    - default           → work_hours_idx=1, health_idx=0, focus_idx=0
""").strip()

SAFE_DEFAULT = {"work_hours_idx": 1, "health_idx": 0, "focus_idx": 0}


# ── Logging helpers (strict format required by validator) ──────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ──────────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs_dict: dict, step: int) -> dict:
    user_prompt = f"Step {step}/90. Current observation:\n{json.dumps(obs_dict, indent=2)}\n\nReturn your action JSON."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=80,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        action = json.loads(text)
        return {
            "work_hours_idx": max(0, min(3, int(action.get("work_hours_idx", 1)))),
            "health_idx":     max(0, min(2, int(action.get("health_idx", 0)))),
            "focus_idx":      max(0, min(3, int(action.get("focus_idx", 0)))),
        }
    except Exception as exc:
        print(f"[DEBUG] LLM error at step {step}: {exc}", flush=True)
        return SAFE_DEFAULT


# ── Single episode runner ─────────────────────────────────────────────────

def run_episode(task_name: str) -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FounderEnvClient(base_url=SPACE_URL, task=task_name)  # ✅ sync client

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        # reset() returns observation object or dict
        obs_dict = result.dict() if hasattr(result, "dict") else dict(result)
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_action(llm_client, obs_dict, step)
            action = FounderAction(**action_dict)
            action_str = json.dumps(action_dict)

            step_result = env.step(action)

            # StepResult has .observation, .reward, .done
            obs = step_result.observation
            obs_dict = obs.dict() if hasattr(obs, "dict") else dict(obs)
            reward  = float(step_result.reward or 0.0)
            done    = bool(step_result.done)
            error   = getattr(step_result, "error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        # Pull final score from state endpoint
        try:
            state = env.state()
            score = float(getattr(state, "score", 0.0))
        except Exception:
            # Fallback: normalise cumulative reward
            raw = sum(rewards)
            score = max(0.0, min(1.0, (raw - (-500)) / (800 - (-500))))

        score   = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point — runs all 3 tasks sequentially ───────────────────────────

if __name__ == "__main__":
    for task in ["survive_easy", "survive_medium", "survive_hard"]:
        run_episode(task)