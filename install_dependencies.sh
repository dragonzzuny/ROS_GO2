#!/bin/bash

echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="

# Install required Python packages
pip install gymnasium
pip install numpy
pip install pyyaml

echo ""
echo "=========================================="
echo "✅ Dependencies installed!"
echo "=========================================="
echo ""
echo "Now you can run tests:"
echo "  python test_industrial_events.py"
echo "  python test_nav2_and_heuristics.py"
