#!/bin/bash
# Phase 3: 3-Stage Curriculum Learning Runner
# Reviewer 박용준

set -e

echo "========================================"
echo "Phase 3: Curriculum Learning"
echo "========================================"
echo ""

# Clear Python cache
echo "🧹 Clearing Python cache..."
find src -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find src -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Cache cleared"
echo ""

# Default parameters
LEARNING_RATE=${1:-3e-4}
START_STAGE=${2:-1}
CUDA_FLAG=""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected - enabling GPU acceleration"
    CUDA_FLAG="--cuda"
else
    echo "💻 No CUDA detected - using CPU"
fi

echo ""
echo "Configuration:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Start Stage: $START_STAGE"
echo "  Device: $([ -n "$CUDA_FLAG" ] && echo "CUDA" || echo "CPU")"
echo ""

# Run curriculum training
echo "🚀 Starting 3-stage curriculum training..."
echo ""

python3 scripts/train_curriculum.py \
  --learning-rate $LEARNING_RATE \
  --start-stage $START_STAGE \
  --num-steps 2048 \
  --num-epochs 10 \
  --batch-size 256 \
  --log-interval 10 \
  --save-interval 50 \
  --experiment-name "curriculum_phase3" \
  --seed 42 \
  $CUDA_FLAG

echo ""
echo "========================================"
echo "✅ Curriculum Training Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Check runs/curriculum_phase3/*/tensorboard for training curves"
echo "  2. Compare stage1 → stage2 → stage3 performance progression"
echo "  3. Verify warm-start transfer from previous stages"
echo "  4. Check success criteria achievement per stage"
echo ""
echo "Expected outcomes:"
echo "  Stage 1 (Simple): Return std <40k, Event success >60%, Coverage >50%"
echo "  Stage 2 (Medium): Return std <40k, Event success >65%, Coverage >55%"
echo "  Stage 3 (Complex): Return std <45k, Event success >60%, Coverage >50%"
echo ""
