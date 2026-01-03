#!/usr/bin/env python3
"""Test if patrol_env.py fixes are loaded correctly"""
import sys
from pathlib import Path

# Add src to path (same as train_multi_map.py)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and check
from rl_dispatch.env.patrol_env import PatrolEnv
import inspect

# Get the step method source
source = inspect.getsource(PatrolEnv.step)

# Check if fix is present
if "Reviewer 박용준: Fix ZeroDivisionError when new_time is 0" in source:
    print("✅ ZeroDivisionError fix is loaded correctly")
else:
    print("❌ ZeroDivisionError fix is NOT found!")
    print("\nSearching for old code...")
    if "expected_visits = new_time / (new_time / self.env_config.num_patrol_points)" in source:
        print("❌ OLD CODE is still being used!")
    else:
        print("⚠️  Cannot find old code either - something is wrong")

if "Reviewer 박용준: Clamp replan_idx to valid range" in source:
    print("✅ replan_idx clamp fix is loaded correctly")
else:
    print("❌ replan_idx clamp fix is NOT found!")

print("\n" + "="*60)
print("File location:", inspect.getfile(PatrolEnv))
print("="*60)
