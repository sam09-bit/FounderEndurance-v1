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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._day = 0
        self._consecutive_overwork = 0
        self._caffeine_clearance_days = 0

        # Initial state -- founder is reasonably healthy at episode start
        obs = np.array([
            0.10,  # sleep_debt
            0.15,  # cortisol_level
            0.00,  # caffeine_toxicity
            0.50,  # product_velocity
            0.80,  # team_morale
            0.60,  # cash_runway
            0.50,  # market_condition (mid-cycle)
            0.00,  # active_crisis
            0.00,  # day_of_week (Monday)
            1.00,  # days_to_launch (full 90 days)
        ], dtype=np.float32)

        self._prev_obs = obs.copy()
        info = {}
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

        self._prev_obs = obs.copy()

        info = {"day": self._day, "action": action.tolist()}

        return obs, float(reward), terminated, truncated, info

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
            # Return a minimal placeholder image
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            return img

    # ------------------------------------------------------------------
    # Dynamics Subsystem 1: Caffeine & Sleep Debt
    # ------------------------------------------------------------------
    def _apply_work_and_sleep(self, obs, work_hours_idx, health_idx):
        hours = self.WORK_HOURS[work_hours_idx]
        using_caffeine = (health_idx == 1)
        using_therapy = (health_idx == 2)

        # Therapy hard-caps work hours at 8
        if using_therapy:
            hours = min(hours, 8)
            obs[1] -= 0.20  # cortisol_level drops sharply

        # Caffeine toxicity
        if using_caffeine:
            obs[2] += 0.20  # caffeine_toxicity
            self._caffeine_clearance_days = 0
        else:
            self._caffeine_clearance_days += 1
            if self._caffeine_clearance_days >= 3:
                obs[2] -= 0.10  # caffeine clears after 3 days

        # Sleep debt accumulation / recovery
        sleep_hours = 24 - hours
        if sleep_hours < 7:
            debt_increase = (7 - sleep_hours) / 7.0 * 0.15
            obs[0] += debt_increase
        else:
            # Can only clear sleep debt if caffeine_toxicity is low
            if obs[2] <= 0.6:
                obs[0] -= 0.05  # gradual recovery

        # Cortisol increases with long hours
        if hours >= 12:
            obs[1] += 0.05 * (hours / 8.0)

        return obs

    # ------------------------------------------------------------------
    # Dynamics Subsystem 2: Focus Area Effects
    # ------------------------------------------------------------------
    def _apply_focus(self, obs, focus_idx, work_hours_idx):
        # Effectiveness multiplier: halved if caffeine_toxicity > 0.6
        effectiveness = 0.5 if obs[2] > 0.6 else 1.0
        hours = self.WORK_HOURS[work_hours_idx]
        intensity = (hours / 16.0) * effectiveness

        if focus_idx == 0:    # Product / Engineering
            obs[3] += 0.08 * intensity    # product_velocity up
            # Crisis resolved faster when team focuses on product
            if obs[7] > 0.0:
                obs[7] -= 0.3 * intensity

        elif focus_idx == 1:  # Fundraising / Sales
            # Market condition scales the cash gain
            cash_gain = 0.10 * intensity * obs[6]
            obs[5] += cash_gain

        elif focus_idx == 2:  # Team Building
            obs[4] += 0.10 * intensity    # team_morale up
            obs[3] -= 0.03               # velocity dips temporarily

        elif focus_idx == 3:  # Firefighting
            obs[1] -= 0.08 * intensity    # cortisol_level down

        # Cash runway always decays by a daily burn rate
        obs[5] -= 0.01

        # Market condition: stochastic sine wave
        # Advance phase by a random small step each day
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
            obs[4] -= decay  # exponential morale loss

        # Hard cap: quiet-quitting kicks in at low morale
        if obs[4] < 0.2:
            obs[3] = min(obs[3], 0.10)  # product_velocity capped at 10%

        return obs

    # ------------------------------------------------------------------
    # Dynamics Subsystem 4: Stochastic Crisis Engine
    # ------------------------------------------------------------------
    def _apply_crisis(self, obs, focus_idx):
        # Crisis resolution first
        if obs[7] > 0.0:
            obs[1] += 0.05  # active crisis raises cortisol each day

        # Stochastic crisis generation
        p_crisis = 0.05 + (1.0 - obs[3]) * 0.15 + obs[2] * 0.10
        if obs[7] == 0.0 and self.np_random.random() < p_crisis:
            obs[7] = 1.0  # new crisis activated

        return obs

    # ------------------------------------------------------------------
    # Reward Function
    # ------------------------------------------------------------------
    def _compute_reward(self, obs):
        prev = self._prev_obs
        reward = 0.0
        terminated = False

        # Delta rewards
        delta_cash = obs[5] - prev[5]
        delta_velocity = obs[3] - prev[3]
        reward += self.W1 * delta_cash
        reward += self.W2 * delta_velocity

        # Health penalties
        if obs[1] > 0.8:    # cortisol_level
            reward -= 1.0
        if obs[0] > 0.8:    # sleep_debt
            reward -= 1.5

        # Terminal: Hospitalisation
        if obs[1] >= 1.0 or obs[0] >= 1.0:
            reward -= 250.0
            terminated = True

        # Terminal: Bankruptcy
        if obs[5] <= 0.0:
            reward -= 250.0
            terminated = True

        # Terminal: Mutiny
        if obs[4] <= 0.0:
            reward -= 200.0
            terminated = True

        return reward, terminated