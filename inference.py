"""
Baseline Inference Script for FounderEnvironment
Strictly follows the [START], [STEP], [END] logging format.
"""
import os
import json
import textwrap
import asyncio
from typing import List, Optional
from openai import OpenAI
from openenv.core.client import OpenEnvClient

# Import local models for typing
from models import FounderAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# OpenEnv sets these env vars during evaluation
TASK_NAME = os.getenv("OPENENV_TASK", "survive_medium")
BENCHMARK = os.getenv("OPENENV_BENCHMARK", "founder-env")
# Read the port/host from env vars usually provided by OpenEnv
SERVER_URL = os.getenv("OPENENV_SERVER_URL", "http://localhost:7860")

MAX_STEPS = 90
TEMPERATURE = 0.2
SUCCESS_SCORE_THRESHOLD = 0.5 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent acting as a Startup Founder. Your goal is to survive 90 days without burning out, running out of cash, or destroying team morale.
    
    You must choose 3 actions every step (day):
    1. work_hours_idx (0=4h, 1=8h, 2=12h, 3=16h)
    2. health_idx (0=Normal, 1=Coffee, 2=Therapy/Rest)
    3. focus_idx (0=Product, 1=Fundraising, 2=Team Building, 3=Crisis Management)
    
    TIPS:
    - If cash is low (<0.3), focus on Fundraising (focus_idx=1).
    - If team morale is low (<0.4), focus on Team Building (focus_idx=2).
    - If cortisol or sleep debt is high (>0.7), work less (work_hours_idx=0 or 1) and use Therapy (health_idx=2).
    - Reply ONLY with a valid JSON object containing exactly these keys: "work_hours_idx", "health_idx", "focus_idx". No other text.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def get_action_from_llm(client: OpenAI, obs: dict) -> dict:
    user_prompt = json.dumps({
        "sleep_debt": round(obs.get("sleep_debt", 0), 2),
        "cortisol": round(obs.get("cortisol_level", 0), 2),
        "cash_runway": round(obs.get("cash_runway", 0), 2),
        "team_morale": round(obs.get("team_morale", 0), 2),
        "active_crisis": round(obs.get("active_crisis", 0), 2)
    })
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=50,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up markdown if model disobeys instructions
        text = text.replace("```json", "").replace("```", "").strip()
        
        parsed = json.loads(text)
        return {
            "work_hours_idx": int(parsed.get("work_hours_idx", 1)),
            "health_idx": int(parsed.get("health_idx", 0)),
            "focus_idx": int(parsed.get("focus_idx", 0))
        }
    except Exception as exc:
        print(f"[DEBUG] Model request failed or parsing error: {exc}", flush=True)
        # Safe fallback action
        return {"work_hours_idx": 1, "health_idx": 0, "focus_idx": 0}

async def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Connect via OpenEnvClient instead of local instantiation 
    # to match standard server-client OpenEnv architecture
    env_client = OpenEnvClient(SERVER_URL)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    
    # Map task name to difficulty for reset
    difficulty = "medium"
    if "easy" in TASK_NAME.lower(): difficulty = "easy"
    elif "hard" in TASK_NAME.lower(): difficulty = "hard"

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Start episode via HTTP API
        response = env_client.reset(options={"difficulty": difficulty})
        obs = response.get("observation", {})
        episode_id = response.get("state", {}).get("episode_id", "")
        
        for step in range(1, MAX_STEPS + 1):
            action_dict = await get_action_from_llm(llm_client, obs)
            action_str = f"work:{action_dict['work_hours_idx']},health:{action_dict['health_idx']},focus:{action_dict['focus_idx']}"
            
            try:
                # Send action to environment
                response = env_client.step(action_dict)
                obs = response.get("observation", {})
                state = response.get("state", {})
                error = None
            except Exception as e:
                error = str(e)
            
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            if done:
                # Update final score from the last state
                score = state.get("score", 0.0)
                break
                
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())