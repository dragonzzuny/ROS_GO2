#!/bin/bash
# Run Phase 3 Test Suite
# Reviewer 박용준

set -e

echo "========================================"
echo "Running Phase 3 Test Suite"
echo "========================================"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run test
python test_phase3_curriculum.py

echo ""
echo "========================================"
echo "Test execution complete"
echo "========================================"
