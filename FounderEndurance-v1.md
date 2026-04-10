FounderEndurance-v1 Technical Documentation

1. Project Overview

FounderEndurance-v1 is a Reinforcement Learning (RL) and Large Language Model (LLM) agent environment built on the OpenEnv framework. It simulates the grueling, high-stakes 90-day journey of a startup founder. Agents must balance product velocity, cash runway, and team morale against the severe physical and mental tolls of burnout, sleep debt, and cortisol accumulation.

The environment tests an agent's ability to perform long-horizon planning, resource management, and crisis mitigation in a highly dynamic, non-stationary environment (e.g., fluctuating market conditions and random crises).

2. Project Structure

Below is the complete file and directory structure of the repository, highlighting the roles of the core engine, configuration files, and validation scripts.

FounderEndurance-v1/
├── .gitattributes
├── .gitignore
├── Dockerfile                      # Containerization instructions for HF Spaces
├── FounderEndurance_v1_Colab.ipynb # Interactive Colab notebook for the environment
├── FounderEndurance_v1_Technical_Doc.docx # Exported Technical documentation
├── README.md                       # Main project documentation
├── client.py                       # Local OpenEnv client wrapper implementation
├── inference.py                    # Baseline LLM agent evaluation script
├── models.py                       # Strictly typed Pydantic models (Actions, Observations, States)
├── openenv.yaml                    # OpenEnv specification and task configurations
├── pyproject.toml                  # Python packaging and setuptools config (handles server mapping)
├── requirements.txt                # Python dependencies list
├── run_validation.py               # Local environment testing script
├── server/                         # Core Environment Logic Module
│   ├── __init__.py
│   ├── app.py                      # FastAPI entry point bridging OpenEnv and the logic engine
│   ├── environment.py              # Simulation mechanics, MDP, and state transitions
│   └── grader.py                   # Fault-tolerant evaluation logic for tasks (Easy, Medium, Hard)
├── test_grader.py                  # Custom stress-test script for validation bot simulation
├── tests/                          # Unit testing suite
│   └── test_env.py                 # Core environment mechanics tests
├── train/                          # RL training scripts
│   └── train_ppo.py                # PPO baseline training script
├── uv.lock                         # Lockfile for the uv package manager
└── validate-submission.sh          # CI/CD and pre-submission validation bash script


3. System Architecture

The project follows the OpenEnv client-server specification, designed for seamless local validation and cloud deployment (e.g., Hugging Face Spaces).

Core Components & Roles

openenv.yaml: The core specification file defining the environment metadata, entry points, and task variations (Easy, Medium, Hard) along with their respective grader mappings.

pyproject.toml: Python packaging configuration. Crucially maps the server entry point and explicitly includes the server/ directory and root modules (models.py, inference.py, grader.py) for cloud installation.

models.py: Defines the strictly typed Pydantic models for the environment (FounderAction, FounderObservation, FounderState).

server/app.py: The FastAPI entry point bridging the OpenEnv framework with the environment logic.

server/environment.py: The core physics and logic engine of the simulation. Handles state transitions, reward shaping, and episode boundaries.

server/grader.py: The fault-tolerant evaluation logic that maps raw simulation scores to deterministic [0.0, 1.0] bounds for leaderboard tracking.

inference.py: The baseline LLM agent script utilizing an OpenAI-compatible client (defaulted to Qwen2.5-72B-Instruct) to play the game, emitting strictly formatted [START], [STEP], and [END] logs.

Dockerfile: Containerization instructions for exposing the FastAPI server on port 7860.

4. Environment Mechanics

The environment executes in a discrete-time MDP (Markov Decision Process) where each step represents 1 day, capping at a maximum of 90 days.

4.1 Observation Space (FounderObservation)

The state is exposed to the agent via a 10-dimensional continuous vector, internally constrained between [0.0, 1.0]:

sleep_debt: Accumulates if sleep < 7 hours. Fatal if >= 1.0.

cortisol_level: Stress indicator. Rises with 12h+ workdays and active crises. Fatal if >= 1.0.

caffeine_toxicity: Rises with coffee consumption. If > 0.6, prevents sleep debt recovery.

product_velocity: Speed of development. Drives final success.

team_morale: Drops if the founder overworks consecutively (>3 days). Fatal if <= 0.0.

cash_runway: Depletes daily. Boosted by fundraising. Fatal if <= 0.0.

market_condition: A sinusoidal curve (sin(step) + 1)/2 representing macroeconomic trends affecting fundraising.

active_crisis: Binary state (0.0 or 1.0). Spikes cortisol and limits progress until managed.

day_of_week: Cyclic temporal feature (0.0 to 1.0).

days_to_launch: Linear countdown from 1.0 to 0.0 (Day 90).

4.2 Action Space (FounderAction)

At each step, the agent submits a discrete, multi-dimensional action composed of three indices:

work_hours_idx (0-3): Determines hours worked (0=4h, 1=8h, 2=12h, 3=16h). Higher hours increase output but heavily penalize sleep and cortisol.

health_idx (0-2):

0 (Normal): Natural caffeine clearance.

1 (Drink Coffee): Spikes caffeine toxicity (+0.20), prevents sleep debt decay, but maximizes hourly output.

2 (Therapy/Rest): Caps work at 8 hours, significantly reduces cortisol (-0.20).

focus_idx (0-3):

0 (Product): Increases product velocity. Mitigates active crises.

1 (Fundraising): Increases cash runway, heavily multiplied by the current market_condition.

2 (Team Building): Restores team morale but slightly reduces product velocity.

3 (Crisis/Burnout Management): Rapidly decreases cortisol.

4.3 State Transition Dynamics

Sleep Mechanics: Sleep = 24 - Work Hours. If sleep < 7, sleep_debt increases.

Intensity Multiplier: Output is scaled by (hours / 16.0). If caffeine toxicity is high (>0.6), output intensity drops by 50% due to "jitters".

Morale Decay: Working > 12 hours increments a consecutive overwork counter. If > 3 days, morale decays exponentially.

Crisis Generation: Base 5% chance of a crisis daily. This probability scales up significantly if product velocity is low or caffeine toxicity is high.

5. Reward Formulation

The reward function is dense, combining immediate delta-based rewards with massive sparse terminal rewards to encourage long-term survival.

Step Reward:
R_t = (W1 * Δcash) + (W2 * Δvelocity) - Health_Penalties

W1 = 1.0 (Cash is king)

W2 = 0.5 (Velocity is important)

Health Penalties: If Cortisol > 0.8, R_t -= 1.0. If Sleep Debt > 0.8, R_t -= 1.5.

Terminal Reward:

Bankruptcy/Burnout (Failure): If Cash <= 0, Morale <= 0, Cortisol >= 1.0, or Sleep Debt >= 1.0, the episode terminates early. R_terminal = -250.0.

Launch (Success): Surviving 90 days with Velocity > 0.8 and Cash > 0.0 results in a massive bonus. R_terminal = +500.0.

Global Scoring:
The cumulative reward is normalized to a 0.0 - 1.0 scale at the end of the episode using the min/max bounds: (cumulative_reward - (-500)) / (800 - (-500)).

6. Tasks and Evaluation (Graders)

To accommodate varying capabilities of LLMs, the environment provides three distinct difficulty tracks via the OpenEnv task specification.

survive_easy:

Init: High Cash (0.8), High Morale (0.9), Favorable Market.

Grader (grade_easy): Highly forgiving. Applies a 1.2x multiplier to the normalized score.

survive_medium:

Init: Average startup conditions (Cash 0.6, Morale 0.8).

Grader (grade_medium): Unaltered, 1:1 mapping of the normalized score.

survive_hard:

Init: Bootstrapped/Crisis mode (Cash 0.3, Morale 0.5, Bear Market 0.2).

Grader (grade_hard): Punishing curve. Applies a 0.9x penalty to the final score.

6.1 Grader Fault-Tolerance

The server/grader.py module is built with aggressive fault tolerance to pass automated OpenEnv validation constraints.

Payload Extraction: Dynamically resolves the score whether OpenEnv injects a pure dictionary, a Pydantic object, or an internal Episode wrapper.

Float Clamping: All return values are strictly wrapped in max(0.0, min(1.0, value)) to guarantee the final score never breaches the OpenEnv specification boundaries, even under negative validation testing.

7. Baseline Agent (inference.py)

The included inference script demonstrates how to interact with the environment via API.

LLM Provider: Configured for Hugging Face inference endpoints (Qwen2.5-72B-Instruct).

Prompt Engineering: Passes the current critical metrics (cash, morale, cortisol, sleep, crisis) as a JSON payload. The system prompt instructs the LLM to output a strict JSON action block.

Fallback Mechanism: If the LLM produces un-parsable output or hits an API rate limit, the script catches the exception and gracefully defaults to a safe action (work: 1, health: 0, focus: 0) to prevent the automated log parser from crashing.

Logging: Implements the strict [START], [STEP], and [END] stdout formatting required by the hackathon validation engines.

8. Deployment Configuration

The environment is packaged for containerized deployment (specifically Hugging Face Spaces Docker environments).

pyproject.toml: Configures setuptools to build the server/ directory and root scripts into the site-packages, bypassing relative-import limitations.

Uvicorn Binding: Binds the FastAPI server to 0.0.0.0:7860 to comply with standard cloud container health checks.