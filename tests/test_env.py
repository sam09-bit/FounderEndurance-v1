# tests/test_env.py
import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import founder_endurance


@pytest.fixture
def env():
    e = gym.make("FounderEndurance-v1")
    yield e
    e.close()


def test_gym_compliance(env):
    """check_env validates spaces, step(), reset() contracts."""
    check_env(env.unwrapped)


def test_reset_returns_valid_obs(env):
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_step_shapes(env):
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)


def test_bankruptcy_terminates(env):
    """Force cash_runway to 0 and verify terminal state."""
    obs, _ = env.reset()
    env.unwrapped._prev_obs[5] = 0.001  # near-zero cash
    # All-nighter + Fundraising in bear market
    for _ in range(5):
        _, _, terminated, _, _ = env.step(np.array([3, 1, 0]))
        if terminated:
            break
    assert terminated


def test_deterministic_with_seed(env):
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)