#!/usr/bin/env python3
"""Test config loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing EnvConfig.load_yaml()...")

from rl_dispatch.core.config import EnvConfig, RewardConfig

# Test loading map config
config_path = Path(__file__).parent / "configs" / "map_large_square.yaml"
print(f"\nLoading: {config_path}")

try:
    env_config = EnvConfig.load_yaml(str(config_path))
    print("✅ EnvConfig loaded successfully!")
    print(f"   Map size: {env_config.map_width}×{env_config.map_height}m")
    print(f"   Patrol points: {env_config.num_patrol_points}")
    print(f"   Robot velocity: {env_config.robot_max_velocity} m/s")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test reward config
try:
    reward_config = RewardConfig.load_yaml(str(config_path))
    print("\n✅ RewardConfig loaded successfully!")
    print(f"   w_event: {reward_config.w_event}")
    print(f"   w_patrol: {reward_config.w_patrol}")
    print(f"   w_safety: {reward_config.w_safety}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All config loading tests passed!")
print("=" * 60)
