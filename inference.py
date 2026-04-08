import asyncio
import os
import json
import textwrap
from typing import List, Optional

import numpy as np
import gymnasium as gym

# Import the environment to register it
try:
    import founder_endurance
except ImportError:
    pass # Assume it's installed or we handle it

from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("FOUNDER_TASK", "founder_survival")
BENCHMARK = os.getenv("FOUNDER_BENCHMARK", "FounderEndurance-v1")
MAX_STEPS = 90
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  # Need halfway good score for success

# Max possible reward is ~500 for launch
MAX_TOTAL_REWARD = 500.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI founder trying to survive 90 days before your startup launches.
    You must balance your health (sleep debt, cortisol, caffeine), your team's morale, your cash runway, and your product velocity. 
    If you hit 0 cash or lose your team, you go bankrupt. If you hit max cortisol or sleep debt, you go to the hospital.
    
    You must output a JSON dictionary with your action each step.
    The action has three components:
    "work_hours_idx": integer (0: 4hrs, 1: 8hrs, 2: 12hrs, 3: 16hrs)
    "focus_idx": integer (0: Product/Eng, 1: Fundraising/Sales, 2: Team Building, 3: Firefighting)
    "health_idx": integer (0: None, 1: Caffeine, 2: Therapy)

    Format your exact entire response perfectly as ONLY valid JSON.
    Example: {"work_hours_idx": 1, "focus_idx": 0, "health_idx": 1}
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: np.ndarray, last_reward: float, history: List[str]) -> str:
    obs_labels = [
        "sleep_debt", "cortisol_level", "caffeine_toxicity",
        "product_velocity", "team_morale", "cash_runway",
        "market_condition", "active_crisis", "day_of_week", "days_to_launch",
    ]
    obs_str = "\n".join([f"- {label}: {val:.3f}" for label, val in zip(obs_labels, obs)])
    history_block = "\n".join(history[-4:]) if history else "None"
    
    return textwrap.dedent(
        f"""
        Day: {step}
        Last reward: {last_reward:.2f}
        Current Observation:
        {obs_str}
        
        Recent Actions:
        {history_block}
        
        Decide your next action (JSON formatted).
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, obs: np.ndarray, last_reward: float, history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Parse JSON
        if text.startswith("```json"):
            text = text[7:-3]
        action_dict = json.loads(text)
        return action_dict
    except Exception as exc:
        print(f"[DEBUG] Model request failed or failed parsing: {exc}", flush=True)
        # Fallback safe action: 8 hours, product, no health mod
        return {"work_hours_idx": 1, "focus_idx": 0, "health_idx": 0}

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize standard gym env instead of docker class for simplicity
    # If the contest explicitly requires OpenEnv validation spec client, swap to that client here
    # For baseline scripts, using gym locally is completely standard.
    env = gym.make("FounderEndurance-v1")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs, info = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            
            action_dict = get_model_message(client, step, obs, last_reward, history)
            action_str = json.dumps(action_dict)
            
            # Map dict to MultiDiscrete array
            action_arr = np.array([
                action_dict.get("work_hours_idx", 1), 
                action_dict.get("focus_idx", 0), 
                action_dict.get("health_idx", 0)
            ])

            obs, reward, terminated, truncated, info = env.step(action_arr)

            done = terminated or truncated
            error = None

            rewards.append(float(reward))
            steps_taken = step
            last_reward = float(reward)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Day {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                # To assign success score based on survival or launch
                if obs[3] > 0.8 and obs[5] > 0.0 and truncated and not terminated:
                    score = 1.0 # Perfect launch
                elif truncated and not terminated:
                    score = 0.5 # Survived, but no launch
                else:
                    score = 0.0 # Failed
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal error in loop: {e}", flush=True)
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
