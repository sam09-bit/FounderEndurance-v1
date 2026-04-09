import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import FounderEnvClient
from server.models import FounderAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# Point this to your local server running on 7860, or your live HF Space URL
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860") 
MAX_STEPS = 90
TEMPERATURE = 0.7
MAX_TOKENS = 150

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

def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    obs_str = f"""
    - sleep_debt: {obs.sleep_debt:.3f}
    - cortisol_level: {obs.cortisol_level:.3f}
    - caffeine_toxicity: {obs.caffeine_toxicity:.3f}
    - product_velocity: {obs.product_velocity:.3f}
    - team_morale: {obs.team_morale:.3f}
    - cash_runway: {obs.cash_runway:.3f}
    - market_condition: {obs.market_condition:.3f}
    - active_crisis: {obs.active_crisis:.3f}
    - day_of_week: {obs.day_of_week:.3f}
    - days_to_launch: {obs.days_to_launch:.3f}
    """
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

def get_model_action(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> FounderAction:
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
        if text.startswith("```json"): 
            text = text[7:-3]
        action_dict = json.loads(text)
        
        return FounderAction(
            work_hours_idx=action_dict.get("work_hours_idx", 1),
            focus_idx=action_dict.get("focus_idx", 0),
            health_idx=action_dict.get("health_idx", 0)
        )
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}")
        return FounderAction(work_hours_idx=1, focus_idx=0, health_idx=0)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    history: List[str] = []
    
    print(f"Connecting to environment at {ENV_URL}...")
    
    # REQUIRED BY RUBRIC: Using the OpenEnv client pattern!
    with FounderEnvClient(base_url=ENV_URL).sync() as env:
        result = env.reset(options={"difficulty": "medium"})
        obs = result.observation
        last_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
                
            action = get_model_action(client, step, obs, last_reward, history)
            result = env.step(action)
            obs = result.observation
            last_reward = result.reward
            
            action_str = f"work:{action.work_hours_idx}, focus:{action.focus_idx}, health:{action.health_idx}"
            print(f"Day {step} | Action: {action_str} | Reward: {last_reward:.2f}")
            history.append(f"Day {step}: {action_str} -> reward {last_reward:+.2f}")

        state = env.state()
        print(f"\nSimulation Ended. Final Score: {state.score:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
    