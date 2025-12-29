# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster training

## Quick Install

```bash
# Navigate to project directory
cd rl_dispatch_mvp

# Install package in development mode
pip install -e .
```

## Install with Dependencies

### For Development (includes testing tools)

```bash
pip install -e ".[dev]"
```

This installs:
- pytest and pytest-cov for testing
- black, flake8, isort for code formatting
- mypy for type checking

### For ROS2 Deployment (future)

```bash
pip install -e ".[ros]"
```

This installs:
- rclpy for ROS2 Python client
- nav2-simple-commander for navigation
- ROS2 message packages

## Verify Installation

```bash
# Test import
python -c "from rl_dispatch.env import PatrolEnv; print('✅ Success!')"

# Run tests
pytest tests/ -v

# Check version
python -c "import rl_dispatch; print(f'Version: {rl_dispatch.__version__}')"
```

## GPU Support

For GPU acceleration (NVIDIA CUDA):

```bash
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Issue: Module not found

```bash
# Ensure you're in the project directory
cd rl_dispatch_mvp

# Reinstall
pip uninstall rl-dispatch
pip install -e .
```

### Issue: Dependency conflicts

```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Next Steps

After installation:

1. **Quick test**: See `QUICK_START.md`
2. **Train model**: `python scripts/train.py`
3. **Run tests**: `pytest tests/`
4. **Read docs**: See `readme/` folder
