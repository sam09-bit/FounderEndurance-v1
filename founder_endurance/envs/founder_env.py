# founder_endurance/envs/founder_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FounderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    # Reward weights (tunable hyperparameters)
    W1 = 1.0   # cash_runway delta weight
    W2 = 0.5   # product_velocity delta weight
    W3 = 2.0   # health penalty weight

    WORK_HOURS = [4, 8, 12, 16]  # indexed by work_hours_idx

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Observation: 10 normalised floats in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Action: [work_hours(4), focus(4), health(3)]
        self.action_space = spaces.MultiDiscrete([4, 4, 3])

        # Internal counters (not part of obs)
        self._obs_labels = [
            "sleep_debt", "cortisol_level", "caffeine_toxicity",
            "product_velocity", "team_morale", "cash_runway",
            "market_condition", "active_crisis", "day_of_week", "days_to_launch",
        ]
        self._consecutive_overwork = 0
        self._caffeine_clearance_days = 0
        self._day = 0
        self._prev_obs = None
        self._cumulative_reward = 0.0 # Track total reward for 0.0-1.0 grader score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._day = 0
        self._consecutive_overwork = 0
        self._caffeine_clearance_days = 0
        self._cumulative_reward = 0.0

        # Handle Difficulty Levels via options dictionary
        options = options or {}
        difficulty = options.get("difficulty", "medium")

        # Set initial stats based on the selected task difficulty
        if difficulty == "easy":
            start_cash, start_morale, start_market = 0.80, 0.90, 0.80
        elif difficulty == "hard":
            start_cash, start_morale, start_market = 0.30, 0.50, 0.20
        else: # medium
            start_cash, start_morale, start_market = 0.60, 0.80, 0.50

        # Initial state
        obs = np.array([
            0.10,  # sleep_debt
            0.15,  # cortisol_level
            0.00,  # caffeine_toxicity
            0.50,  # product_velocity
            start_morale,  
            start_cash,  
            start_market, 
            0.00,  # active_crisis
            0.00,  # day_of_week (Monday)
            1.00,  # days_to_launch (full 90 days)
        ], dtype=np.float32)

        self._prev_obs = obs.copy()
        info = {"difficulty": difficulty}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        work_hours_idx, focus_idx, health_idx = action
        obs = self._prev_obs.copy()

        # 1. Apply work-hours effect on sleep_debt
        obs = self._apply_work_and_sleep(obs, work_hours_idx, health_idx)

        # 2. Apply focus-area effect on company variables
        obs = self._apply_focus(obs, focus_idx, work_hours_idx)

        # 3. Apply death-march morale decay (non-linear)
        obs = self._apply_morale_decay(obs, work_hours_idx)

        # 4. Stochastic crisis generation / resolution
        obs = self._apply_crisis(obs, focus_idx)

        # 5. Advance time variables
        self._day += 1
        obs[8] = (self._day % 7) / 7.0
        obs[9] = max(0.0, (90 - self._day) / 90.0)

        # 6. Clip all values to [0, 1]
        obs = np.clip(obs, 0.0, 1.0)

        # 7. Compute reward
        reward, terminated = self._compute_reward(obs)

        truncated = self._day >= 90

        if truncated and not terminated:
            # Check for successful launch
            if obs[3] > 0.8 and obs[5] > 0.0:
                reward += 500.0

        self._cumulative_reward += reward
        self._prev_obs = obs.copy()

        # Compile info dict
        info = {"day": self._day, "action": action.tolist()}
        
        # If episode is over, calculate the normalized 0.0 - 1.0 grader score
        if terminated or truncated:
            info["score"] = self._calculate_normalized_score(self._cumulative_reward)

        return obs, float(reward), terminated, truncated, info

    def _calculate_normalized_score(self, total_reward):
        """Maps the raw cumulative reward to a strict 0.0 to 1.0 scale for the OpenEnv Grader."""
        # Estimated bounds based on penalties and launch bonuses
        MIN_REWARD = -500.0 
        MAX_REWARD = 800.0
        
        score = (total_reward - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
        return float(max(0.0, min(1.0, score)))

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            if self._prev_obs is not None:
                print(f"Day {self._day}")
                for i, label in enumerate(self._obs_labels):
                    print(f"  {label}: {self._prev_obs[i]:.3f}")
            return None
        if self.render_mode == "rgb_array":
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            return img

    # ------------------------------------------------------------------
    # Dynamics Subsystem 1: Caffeine & Sleep Debt
    # ------------------------------------------------------------------
    def _apply_work_and_sleep(self, obs, work_hours_idx, health_idx):
        hours = self.WORK_HOURS[work_hours_idx]
        using_caffeine = (health_idx == 1)
        using_therapy = (health_idx == 2)

        if using_therapy:
            hours = min(hours, 8)
            obs[1] -= 0.20  

        if using_caffeine:
            obs[2] += 0.20  
            self._caffeine_clearance_days = 0
        else:
            self._caffeine_clearance_days += 1
            if self._caffeine_clearance_days >= 3:
                obs[2] -= 0.10  

        sleep_hours = 24 - hours
        if sleep_hours < 7:
            debt_increase = (7 - sleep_hours) / 7.0 * 0.15
            obs[0] += debt_increase
        else:
            if obs[2] <= 0.6:
                obs[0] -= 0.05  

        if hours >= 12:
            obs[1] += 0.05 * (hours / 8.0)

        return obs

    # ------------------------------------------------------------------
    # Dynamics Subsystem 2: Focus Area Effects
    # ------------------------------------------------------------------
    def _apply_focus(self, obs, focus_idx, work_hours_idx):
        effectiveness = 0.5 if obs[2] > 0.6 else 1.0
        hours = self.WORK_HOURS[work_hours_idx]
        intensity = (hours / 16.0) * effectiveness

        if focus_idx == 0:    
            obs[3] += 0.08 * intensity    
            if obs[7] > 0.0:
                obs[7] -= 0.3 * intensity

        elif focus_idx == 1:  
            cash_gain = 0.10 * intensity * obs[6]
            obs[5] += cash_gain

        elif focus_idx == 2:  
            obs[4] += 0.10 * intensity    
            obs[3] -= 0.03               

        elif focus_idx == 3:  
            obs[1] -= 0.08 * intensity    

        obs[5] -= 0.01

        phase_delta = self.np_random.uniform(0.02, 0.06)
        obs[6] = (np.sin(self._day * phase_delta) + 1.0) / 2.0

        return obs

    # ------------------------------------------------------------------
    # Dynamics Subsystem 3: Death March (Non-Linear Morale Decay)
    # ------------------------------------------------------------------
    def _apply_morale_decay(self, obs, work_hours_idx):
        hours = self.WORK_HOURS[work_hours_idx]

        if hours > 12:
            self._consecutive_overwork += 1
        else:
            self._consecutive_overwork = max(0, self._consecutive_overwork - 1)

        if self._consecutive_overwork > 3:
            decay = 0.05 * (self._consecutive_overwork ** 1.5)
            obs[4] -= decay  

        if obs[4] < 0.2:
            obs[3] = min(obs[3], 0.10)  

        return obs

    # ------------------------------------------------------------------
    # Dynamics Subsystem 4: Stochastic Crisis Engine
    # ------------------------------------------------------------------
    def _apply_crisis(self, obs, focus_idx):
        if obs[7] > 0.0:
            obs[1] += 0.05  

        p_crisis = 0.05 + (1.0 - obs[3]) * 0.15 + obs[2] * 0.10
        if obs[7] == 0.0 and self.np_random.random() < p_crisis:
            obs[7] = 1.0  

        return obs

    # ------------------------------------------------------------------
    # Reward Function
    # ------------------------------------------------------------------
    def _compute_reward(self, obs):
        prev = self._prev_obs
        reward = 0.0
        terminated = False

        delta_cash = obs[5] - prev[5]
        delta_velocity = obs[3] - prev[3]
        reward += self.W1 * delta_cash
        reward += self.W2 * delta_velocity

        if obs[1] > 0.8:    
            reward -= 1.0
        if obs[0] > 0.8:    
            reward -= 1.5

        if obs[1] >= 1.0 or obs[0] >= 1.0:
            reward -= 250.0
            terminated = True

        if obs[5] <= 0.0:
            reward -= 250.0
            terminated = True

        if obs[4] <= 0.0:
            reward -= 200.0
            terminated = True

        return reward, terminated