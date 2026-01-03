#!/bin/bash
# Phase 2 Validation Training Script
# Reviewer 박용준: Short training run to validate Phase 2 improvements

set -e

echo "=================================="
echo "Phase 2 Validation Training"
echo "=================================="
echo ""

# Quick sanity check - clear Python cache
echo "🧹 Clearing Python cache..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find src -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Cache cleared"
echo ""

# Run short training on campus map (worst-case scenario)
echo "🚀 Starting validation training..."
echo "   Map: campus (worst-case before Phase 2)"
echo "   Steps: 100,000 (quick validation)"
echo "   Expected: Return std < 40k (was 83k before)"
echo ""

python3 scripts/train.py \
  --config configs/map_campus.yaml \
  --total_timesteps 100000 \
  --checkpoint_dir checkpoints/phase2_validation \
  --tensorboard_log logs/phase2_validation \
  --save_freq 20000 \
  --verbose 1

echo ""
echo "=================================="
echo "✅ Validation training complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Check logs/phase2_validation in TensorBoard"
echo "  2. Verify reward components are balanced"
echo "  3. Check Return mean and std improvement"
echo "  4. Look for 'rollout/ep_rew_mean' trend"
echo ""
echo "Expected improvements:"
echo "  ❌ Before Phase 2: Return std ~83k, nav failure 95%"
echo "  ✅ After Phase 2:  Return std <40k, nav failure <10%"
echo ""
