#!/bin/bash
# Phase 1 Implementation Script
# Applies all necessary patches for feasible goal generation

set -e

echo "=================================="
echo "Phase 1 Implementation Script"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "src/rl_dispatch/env/patrol_env.py" ]; then
    echo "❌ Error: Not in rl_dispatch_mvp directory"
    exit 1
fi

echo "📝 Step 1: Patching patrol_env.py..."

# Create backup
cp src/rl_dispatch/env/patrol_env.py src/rl_dispatch/env/patrol_env.py.phase1_backup

# Add the connection line after nav_interface initialization
python3 << 'EOF'
with open('src/rl_dispatch/env/patrol_env.py', 'r') as f:
    content = f.read()

# Find the nav_interface initialization block
search_text = """        self.nav_interface = SimulatedNav2(
            occupancy_grid=self.occupancy_grid,
            grid_resolution=self.env_config.grid_resolution,
            max_velocity=self.env_config.robot_max_velocity,
            nav_failure_rate=0.05,
            collision_rate=0.01,
            np_random=np.random.RandomState(seed),
        )

        # Initialize robot at random patrol point"""

replace_text = """        self.nav_interface = SimulatedNav2(
            occupancy_grid=self.occupancy_grid,
            grid_resolution=self.env_config.grid_resolution,
            max_velocity=self.env_config.robot_max_velocity,
            nav_failure_rate=0.05,
            collision_rate=0.01,
            np_random=np.random.RandomState(seed),
        )

        # Reviewer 박용준: Connect nav_interface to candidate factory for A* pathfinding
        self.candidate_factory.set_nav_interface(self.nav_interface)

        # Initialize robot at random patrol point"""

if 'candidate_factory.set_nav_interface' in content:
    print("✅ patrol_env.py already patched")
elif search_text in content:
    content = content.replace(search_text, replace_text)
    with open('src/rl_dispatch/env/patrol_env.py', 'w') as f:
        f.write(content)
    print("✅ patrol_env.py patched successfully")
else:
    print("⚠️  Could not find exact match - manual edit required")
    print("   Please add this line after nav_interface initialization:")
    print("   self.candidate_factory.set_nav_interface(self.nav_interface)")
EOF

echo ""
echo "📝 Step 2: Checking candidate_generator.py..."

if grep -q "def set_nav_interface" src/rl_dispatch/planning/candidate_generator.py 2>/dev/null; then
    echo "✅ candidate_generator.py already has set_nav_interface method"
else
    echo "⚠️  candidate_generator.py needs manual update"
    echo "   See PHASE1_PATCHES.md for detailed instructions"
    echo "   Or copy the complete new version from the patches"
fi

echo ""
echo "=================================="
echo "✅ Phase 1 patches applied!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. If candidate_generator.py needs update, apply manually"
echo "  2. Run: python test_phase1_feasible_goals.py"
echo "  3. Verify all tests pass"
echo ""
