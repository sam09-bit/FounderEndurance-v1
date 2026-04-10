import sys
import os
# Add the root directory to sys.path so we can import models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import uuid
import random
from openenv.core.env_server import Environment
from models import FounderAction, FounderObservation, FounderState

class FounderEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    W1 = 1.0   
    W2 = 0.5   
    WORK_HOURS = [4, 8, 12, 16]

    def __init__(self):
        self._state = FounderState()
        self._obs_array = np.zeros(10, dtype=np.float32)
        self._consecutive_overwork = 0
        self._caffeine_clearance_days = 0
        self._cumulative_reward = 0.0

    def reset(self, episode_id: str = None, seed: int = None, options: dict = None, **kwargs) -> FounderObservation:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self._consecutive_overwork = 0
        self._caffeine_clearance_days = 0
        self._cumulative_reward = 0.0

        # Bulletproof options extraction
        if options is None:
            options = kwargs.get("options", {})
        if not isinstance(options, dict):
            options = {}
            
        difficulty = options.get("difficulty", "medium")
        
        if difficulty == "easy":
            start_cash, start_morale, start_market = 0.80, 0.90, 0.80
        elif difficulty == "hard":
            start_cash, start_morale, start_market = 0.30, 0.50, 0.20
        else: 
            start_cash, start_morale, start_market = 0.60, 0.80, 0.50

        self._state = FounderState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty,
            score=0.0
        )

        self._obs_array = np.array([
            0.10, 0.15, 0.00, 0.50, start_morale, start_cash, start_market, 0.00, 0.00, 1.00
        ], dtype=np.float32)

        return self._array_to_obs(done=False, reward=0.0)

    def step(self, action: FounderAction, timeout_s=None, **kwargs) -> FounderObservation:
        self._state.step_count += 1
        prev_obs = self._obs_array.copy()
        
        # 1. Work & Sleep
        hours = self.WORK_HOURS[action.work_hours_idx]
        if action.health_idx == 2:
            hours = min(hours, 8)
            self._obs_array[1] -= 0.20  
        if action.health_idx == 1:
            self._obs_array[2] += 0.20  
            self._caffeine_clearance_days = 0
        else:
            self._caffeine_clearance_days += 1
            if self._caffeine_clearance_days >= 3: self._obs_array[2] -= 0.10  

        sleep_hours = 24 - hours
        if sleep_hours < 7:
            self._obs_array[0] += ((7 - sleep_hours) / 7.0) * 0.15
        elif self._obs_array[2] <= 0.6:
            self._obs_array[0] -= 0.05  
        if hours >= 12: self._obs_array[1] += 0.05 * (hours / 8.0)

        # 2. Focus
        intensity = (hours / 16.0) * (0.5 if self._obs_array[2] > 0.6 else 1.0)
        if action.focus_idx == 0:    
            self._obs_array[3] += 0.08 * intensity    
            if self._obs_array[7] > 0.0: self._obs_array[7] -= 0.3 * intensity
        elif action.focus_idx == 1:  
            self._obs_array[5] += 0.10 * intensity * self._obs_array[6]
        elif action.focus_idx == 2:  
            self._obs_array[4] += 0.10 * intensity    
            self._obs_array[3] -= 0.03               
        elif action.focus_idx == 3:  
            self._obs_array[1] -= 0.08 * intensity    

        self._obs_array[5] -= 0.01
        self._obs_array[6] = (np.sin(self._state.step_count * np.random.uniform(0.02, 0.06)) + 1.0) / 2.0

        # 3. Morale Decay
        self._consecutive_overwork = self._consecutive_overwork + 1 if hours > 12 else max(0, self._consecutive_overwork - 1)
        if self._consecutive_overwork > 3: self._obs_array[4] -= 0.05 * (self._consecutive_overwork ** 1.5)
        if self._obs_array[4] < 0.2: self._obs_array[3] = min(self._obs_array[3], 0.10)

        # 4. Crisis
        if self._obs_array[7] > 0.0: self._obs_array[1] += 0.05  
        if self._obs_array[7] == 0.0 and np.random.random() < (0.05 + (1.0 - self._obs_array[3]) * 0.15 + self._obs_array[2] * 0.10):
            self._obs_array[7] = 1.0  

        # 5. Time
        self._obs_array[8] = (self._state.step_count % 7) / 7.0
        self._obs_array[9] = max(0.0, (90 - self._state.step_count) / 90.0)

        self._obs_array = np.clip(self._obs_array, 0.0, 1.0)

        # 6. Reward & Termination
        reward = self.W1 * (self._obs_array[5] - prev_obs[5]) + self.W2 * (self._obs_array[3] - prev_obs[3])
        if self._obs_array[1] > 0.8: reward -= 1.0
        if self._obs_array[0] > 0.8: reward -= 1.5

        terminated = False
        if self._obs_array[1] >= 1.0 or self._obs_array[0] >= 1.0 or self._obs_array[5] <= 0.0 or self._obs_array[4] <= 0.0:
            reward -= 250.0
            terminated = True

        truncated = self._state.step_count >= 90
        if truncated and not terminated and self._obs_array[3] > 0.8 and self._obs_array[5] > 0.0:
            reward += 500.0

        self._cumulative_reward += reward
        done = terminated or truncated

        if done:
            # We just save the raw score to state. 
            # OpenEnv will automatically pass this state to grader.py based on openenv.yaml!
            raw_score = float(max(0.0, min(1.0, (self._cumulative_reward - (-500.0)) / (800.0 - (-500.0)))))
            self._state.score = raw_score

        return self._array_to_obs(done, float(reward))

    @property
    def state(self) -> FounderState:
        return self._state

    def _array_to_obs(self, done: bool, reward: float) -> FounderObservation:
        return FounderObservation(
            done=done,
            reward=reward,
            sleep_debt=float(self._obs_array[0]),
            cortisol_level=float(self._obs_array[1]),
            caffeine_toxicity=float(self._obs_array[2]),
            product_velocity=float(self._obs_array[3]),
            team_morale=float(self._obs_array[4]),
            cash_runway=float(self._obs_array[5]),
            market_condition=float(self._obs_array[6]),
            active_crisis=float(self._obs_array[7]),
            day_of_week=float(self._obs_array[8]),
            days_to_launch=float(self._obs_array[9])
        )