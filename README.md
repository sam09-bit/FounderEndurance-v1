---
title: FounderEndurance V1
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FounderEndurance-v1: OpenEnv Submission

FounderEndurance-v1 is a startup-founder survival simulator RL environment built for the OpenEnv Hackathon.

## Frontier Model Challenge Justification

Our hard task ("Survive 90 days as a startup founder in a highly volatile market") is designed to genuinely challenge state-of-the-art LLM reasoning models. It does this through:

* **Delayed Consequences:** If an agent chooses an action like `work_hours=16`, `sleep_debt` and `cortisol` slowly accumulate, but the terminal penalty (hospitalization) doesn't hit until days later. LLMs are notoriously bad at attributing delayed negative consequences to past actions without strict chain-of-thought tracking.
* **Non-Linear Dynamics:** The `morale_decay` mechanic utilizes an exponential curve (`decay = 0.05 * (consecutive_overwork ** 1.5)`). LLMs typically assume linear progression and frequently fail to predict the sudden collapse of team morale (mutiny).
* **Long Time Horizons:** The episode spans 90 steps. Autoregressive LLM agents suffer from "drifting attention" over long rollouts and struggle to maintain a coherent fundraising/product strategy for 90 consecutive turns.
* **Harsh Initial State:** In the `hard` difficulty setting, the agent begins with `0.30` cash in a bearish `0.20` market, leaving a 0% margin for error and forcing complex, multi-variable balancing that breaks simple heuristic approaches.