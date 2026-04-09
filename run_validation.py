import yaml
import sys
import gymnasium as gym
import founder_endurance # This registers your environment

def validate_submission():
    print("Starting OpenEnv Validation...\n")
    
    # 1. Check openenv.yaml
    try:
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        print("✅ openenv.yaml loaded successfully.")
        
        # Check tasks
        tasks = config.get("tasks", [])
        if len(tasks) >= 3:
            print(f"✅ Found {len(tasks)} tasks (Meets 3+ requirement).")
        else:
            print(f"❌ Found only {len(tasks)} tasks. You need at least 3.")
            
    except FileNotFoundError:
        print("❌ Could not find openenv.yaml")
        sys.exit(1)
        
    # 2. Check Gymnasium Environment
    try:
        env_id = config["benchmark"]["name"]
        print(f"\nTesting environment: {env_id}")
        
        env = gym.make(env_id)
        print(f"✅ Environment initialized successfully.")
        
        # Test Reset
        obs, info = env.reset()
        print(f"✅ Reset successful. Obs shape: {obs.shape}")
        
        # Test Step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful. Reward type: {type(reward)}")
        
        # Check normalized score in info
        if "score" in info:
            print(f"✅ Grader score found in info dictionary: {info['score']}")
            if 0.0 <= info["score"] <= 1.0:
                 print("✅ Grader score is correctly between 0.0 and 1.0.")
            else:
                 print(f"❌ Grader score {info['score']} is outside 0.0 - 1.0 range!")
        else:
            print("⚠️ 'score' not found in info dict after step. (It might only appear on termination, which is fine).")
            
        env.close()
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        
    print("\nValidation complete. Please check the output above.")

if __name__ == "__main__":
    validate_submission()