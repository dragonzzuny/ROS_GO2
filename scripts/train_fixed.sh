#!/bin/bash

echo "🔧 Preparing training environment..."

# Clear all Python caches
echo "1. Clearing Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Verify the fix is in place
echo "2. Verifying patrol_env.py fixes..."
if grep -q "Reviewer 박용준: Fix ZeroDivisionError when new_time is 0" src/rl_dispatch/env/patrol_env.py; then
    echo "   ✅ ZeroDivisionError fix verified"
else
    echo "   ❌ Fix not found!"
    exit 1
fi

if grep -q "Reviewer 박용준: Clamp replan_idx to valid range" src/rl_dispatch/env/patrol_env.py; then
    echo "   ✅ replan_idx clamp fix verified"
else
    echo "   ❌ Fix not found!"
    exit 1
fi

# Set PYTHONDONTWRITEBYTECODE to prevent .pyc creation
export PYTHONDONTWRITEBYTECODE=1

echo "3. Starting training..."
echo ""

# Run with python3 explicitly
exec python3 scripts/train_multi_map.py "$@"
