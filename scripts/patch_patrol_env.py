#!/usr/bin/env python3
"""
Patch patrol_env.py to connect nav_interface to candidate_factory
"""

import sys

def main():
    filepath = 'src/rl_dispatch/env/patrol_env.py'

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    # Check if already patched
    if 'candidate_factory.set_nav_interface' in content:
        print("✅ patrol_env.py already patched - no changes needed")
        return True

    # Find the pattern to replace
    search_pattern = """        )

        # Initialize robot at random patrol point"""

    replace_pattern = """        )

        # Reviewer 박용준: Connect nav_interface to candidate factory for A* pathfinding
        self.candidate_factory.set_nav_interface(self.nav_interface)

        # Initialize robot at random patrol point"""

    if search_pattern not in content:
        print("❌ Could not find insertion point")
        print("   Manual edit required - see PHASE1_PATCHES.md")
        return False

    # Create backup
    try:
        with open(filepath + '.phase1_backup', 'w', encoding='utf-8') as f:
            f.write(content)
        print("📝 Created backup: patrol_env.py.phase1_backup")
    except Exception as e:
        print(f"⚠️  Could not create backup: {e}")

    # Apply patch
    new_content = content.replace(search_pattern, replace_pattern, 1)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ patrol_env.py patched successfully!")
        print("   Added: self.candidate_factory.set_nav_interface(self.nav_interface)")
        return True
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
