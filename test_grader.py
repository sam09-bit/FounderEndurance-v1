import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Safely import the graders, checking both possible locations
try:
    from grader import grade_easy, grade_medium, grade_hard
    print("✅ Found grader.py in ROOT directory.")
except ModuleNotFoundError:
    try:
        from server.grader import grade_easy, grade_medium, grade_hard
        print("✅ Found grader.py in SERVER/ directory.")
    except ModuleNotFoundError:
        print("❌ ERROR: Could not find grader.py anywhere!")
        sys.exit(1)

from models import FounderState

# Dummy Episode class to simulate OpenEnv's internal wrapper
class DummyEpisode:
    def __init__(self, state_data):
        self.state = state_data

def run_strict_validation():
    print("\n--- Starting OpenEnv Grader Stress Test ---")
    
    graders = {
        "Easy": grade_easy,
        "Medium": grade_medium,
        "Hard": grade_hard
    }

    # The 8 extreme scenarios the validator will try to use to break your code
    test_scenarios = [
        ("1. Dictionary", {"score": 0.75}),
        ("2. Pydantic Object", FounderState(score=0.75)),
        ("3. Episode Wrapper (Dict)", DummyEpisode({"score": 0.75})),
        ("4. Episode Wrapper (Pydantic)", DummyEpisode(FounderState(score=0.75))),
        ("5. Extreme Negative Bound", {"score": -250.0}),
        ("6. Extreme Positive Bound", {"score": 500.0}),
        ("7. Garbage/Empty Data", {}),
        ("8. None Type", None)
    ]

    all_passed = True

    for grader_name, grader_func in graders.items():
        print(f"\n🧪 Testing {grader_name} Grader:")
        
        for scenario_name, payload in test_scenarios:
            try:
                # Call the grader just like the OpenEnv validator does
                result = grader_func(state=payload)
                
                # Check 1: Did it return a float?
                if not isinstance(result, float):
                    print(f"  [FAIL] {scenario_name}: Returned {type(result)} instead of float.")
                    all_passed = False
                    continue
                
                # Check 2: Is it strictly between 0.0 and 1.0?
                if result < 0.0 or result > 1.0:
                    print(f"  [FAIL] {scenario_name}: Returned {result} (Must be exactly between 0.0 and 1.0).")
                    all_passed = False
                    continue
                
                print(f"  [PASS] {scenario_name} -> Output: {result:.3f}")
                
            except Exception as e:
                print(f"  [FAIL] {scenario_name}: Grader crashed with error -> {str(e)}")
                all_passed = False
    print("\n" + "="*45)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your grader is structurally flawless.")
        print("The online 'Task Validation' has no mathematical reason to fail this.")
    else:
        print("❌ GRADER FAILED. Please paste the failures back to me!")
    print("="*45 + "\n")

if __name__ == "__main__":
    run_strict_validation()