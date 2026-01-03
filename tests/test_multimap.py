#!/usr/bin/env python3
"""Quick test script for multi-map system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("Multi-Map System Verification Test")
print("=" * 80)

# Test 1: Import
print("\n[1/5] Testing imports...")
try:
    from rl_dispatch.env import MultiMapPatrolEnv, create_multi_map_env
    from rl_dispatch.core.config import EnvConfig
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load map configs
print("\n[2/5] Loading map configurations...")
try:
    config_dir = Path(__file__).parent / "configs"
    map_configs = [
        str(config_dir / "map_large_square.yaml"),
        str(config_dir / "map_corridor.yaml"),
        str(config_dir / "map_l_shaped.yaml"),
        str(config_dir / "map_office_building.yaml"),
        str(config_dir / "map_campus.yaml"),
        str(config_dir / "map_warehouse.yaml"),
    ]

    for config_path in map_configs:
        if not Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            sys.exit(1)

    print(f"✅ Found all {len(map_configs)} map configs")
except Exception as e:
    print(f"❌ Config loading failed: {e}")
    sys.exit(1)

# Test 3: Create environment
print("\n[3/5] Creating MultiMapPatrolEnv...")
try:
    env = create_multi_map_env(
        map_configs=map_configs,
        track_coverage=True,
    )
    print(f"✅ Environment created")
    print(f"   Maps loaded: {len(env.map_names)}")
    for name in env.map_names:
        config = env.env_configs[name]
        print(f"   - {name}: {config.map_width}×{config.map_height}m, {config.num_patrol_points} points")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Reset and check
print("\n[4/5] Testing environment reset...")
try:
    obs, info = env.reset()
    print(f"✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Selected map: {info['map_name']}")
    print(f"   Episode count: {info['episode_count']}")
except Exception as e:
    print(f"❌ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Run a few steps
print("\n[5/5] Running test episodes...")
try:
    for ep in range(3):
        obs, info = env.reset()
        map_name = info['map_name']
        done = False
        steps = 0

        while not done and steps < 10:  # Limit to 10 steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        print(f"   Episode {ep+1}: {map_name}, {steps} steps, reward={reward:.2f}")

    print("✅ Episodes ran successfully")
except Exception as e:
    print(f"❌ Episode execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check coverage tracking
print("\n[6/6] Checking coverage tracking...")
try:
    heatmaps = env.get_all_coverage_heatmaps()
    print(f"✅ Coverage tracking works")
    for map_name, heatmap in heatmaps.items():
        visits = heatmap.sum()
        print(f"   {map_name}: {heatmap.shape}, {int(visits)} total visits")
except Exception as e:
    print(f"❌ Coverage tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nMulti-Map System is working correctly!")
print("\nNext steps:")
print("  1. Train: python scripts/train_multi_map.py --total-timesteps 100000")
print("  2. Visualize: python scripts/visualize_coverage.py --episodes-per-map 5")
print("=" * 80)

env.close()
