#!/bin/bash

echo "=========================================="
echo "Installing Training Dependencies"
echo "=========================================="

# Install PyTorch (CPU version for compatibility)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
echo ""
echo "Installing additional packages..."
pip install tensorboard
pip install matplotlib
pip install tqdm

echo ""
echo "=========================================="
echo "✅ Training dependencies installed!"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - torch (PyTorch)"
echo "  - tensorboard (logging)"
echo "  - matplotlib (visualization)"
echo "  - tqdm (progress bars)"
echo ""
echo "Now you can start training:"
echo "  python scripts/train_multi_map.py --total-timesteps 100000"
