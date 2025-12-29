# Contributing to RL Dispatch MVP

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/rl_dispatch_mvp.git
cd rl_dispatch_mvp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
pytest tests/ -v
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following PEP 8 style guide
- Add type hints to all functions
- Write docstrings (Google style)
- Add unit tests for new functionality

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Check formatting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: descriptive commit message"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use `black` for formatting (line length: 100)
- Use `isort` for import sorting
- Maximum line length: 100 characters
- Use type hints for all function parameters and returns

### Documentation Style

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Short description of function.

    Longer description if needed. Explain what the function does,
    any important details, and edge cases.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Returns:
        Description of return value

    Raises:
        ValueError: When parameter is invalid

    Example:
        >>> example_function(42, "hello")
        True
    """
    pass
```

### Testing Guidelines

- Write unit tests for all new functions
- Use pytest fixtures for common setup
- Aim for >85% code coverage
- Test edge cases and error conditions

Example test:

```python
def test_example_function():
    """Test that example_function works correctly."""
    result = example_function(42, "hello")
    assert result == True

def test_example_function_invalid_input():
    """Test that example_function raises error for invalid input."""
    with pytest.raises(ValueError):
        example_function(-1, "")
```

## Project Structure

When adding new features, place them in the appropriate module:

- `src/rl_dispatch/core/` - Core data structures and configs
- `src/rl_dispatch/env/` - Environment implementation
- `src/rl_dispatch/algorithms/` - RL algorithms and policies
- `src/rl_dispatch/planning/` - Route planning strategies
- `src/rl_dispatch/rewards/` - Reward calculation
- `src/rl_dispatch/utils/` - Utility functions
- `scripts/` - Executable scripts
- `tests/` - Unit and integration tests

## What to Contribute

### High Priority
- Bug fixes
- Performance improvements
- Documentation improvements
- Additional unit tests
- Example notebooks/tutorials

### Medium Priority
- New baseline policies
- Visualization improvements
- Additional candidate strategies
- Hyperparameter tuning tools

### Low Priority
- Code refactoring
- Style improvements
- Minor optimizations

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal code to reproduce the problem
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version
   - PyTorch version
   - OS and version
   - GPU/CUDA version (if applicable)

Example:

```
**Description**
Training fails with CUDA out of memory error

**Steps to Reproduce**
```python
python scripts/train.py --cuda --batch-size 1024
```

**Expected Behavior**
Training should run successfully

**Actual Behavior**
RuntimeError: CUDA out of memory

**Environment**
- Python 3.9
- PyTorch 2.0.1
- Ubuntu 20.04
- NVIDIA RTX 3090, CUDA 11.7
```

## Pull Request Process

1. **Update Documentation**: Update README, docstrings, and comments
2. **Add Tests**: Ensure new code has adequate test coverage
3. **Run Tests**: All tests must pass before PR
4. **Update CHANGELOG**: Add entry describing changes
5. **Request Review**: Tag maintainers for review

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No merge conflicts
- [ ] Commit messages are descriptive

## Code Review Process

Maintainers will review PRs for:
- Code quality and style
- Test coverage
- Documentation
- Performance implications
- Backward compatibility

Please be patient - reviews may take a few days.

## Community Guidelines

- Be respectful and constructive
- Help others learn
- Follow the code of conduct
- Ask questions if unclear
- Credit others' work

## Questions?

If you have questions:
- Open a GitHub Discussion
- Check existing issues
- Read the documentation in `readme/`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
