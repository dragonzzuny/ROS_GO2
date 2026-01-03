"""Quick import test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")
from rl_dispatch.env import MultiMapPatrolEnv, create_multi_map_env
print("SUCCESS: MultiMapPatrolEnv imported!")

from rl_dispatch.core.config import EnvConfig
print("SUCCESS: EnvConfig imported!")

print("\nAll imports successful! ✅")
