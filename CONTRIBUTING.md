# Contributing to ExtraSensory Transfer Entropy Analysis

We welcome contributions! This document provides guidelines for contributing to this project.

## Code of Conduct

This project adheres to a code of professional conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check existing [GitHub Issues](https://github.com/yourusername/extrasensory_analysis/issues)
2. Ensure you're using the latest version
3. Test with a minimal reproducible example

**Bug Report Template**:
```markdown
**Environment**:
- Python version: [e.g., 3.12.1]
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 14]
- JIDT version: [found in run_info.yaml]

**Config**:
```yaml
# Paste relevant config/template.yaml (reference) or config/presets/ sections
```

**Steps to Reproduce**:
1. Run command: `python run_production.py --smoke`
2. Observe error in line X of errors.log
3. ...

**Expected Behavior**: [What should happen]
**Actual Behavior**: [What actually happens]
**Error Log**: [Paste from errors.log]
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub Issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **List any alternative solutions** you've considered

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding style** (see Style Guide below)
3. **Add tests** for new functionality
4. **Update documentation** (README.md, config/README.md, docstrings)
5. **Ensure all tests pass** before submitting
6. **Write a clear commit message**

**PR Template**:
```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change required? What problem does it solve?

## How Has This Been Tested?
Describe testing:
- [ ] Smoke test passes
- [ ] Full 60-user run tested
- [ ] Unit tests added/updated

## Types of Changes
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Checklist
- [ ] Code follows project style guidelines
- [ ] Added tests that prove fix/feature works
- [ ] Updated documentation
- [ ] All tests pass
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/extrasensory_analysis.git
cd extrasensory_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Run smoke test
python run_production.py --smoke
```

## Style Guide

### Python Code Style

**Follow PEP 8** with these specifics:
- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group stdlib, third-party, local (separated by blank lines)
- **Docstrings**: NumPy style for functions/classes

**Use Black for formatting**:
```bash
black src/ tests/ run_production.py
```

**Run linting**:
```bash
flake8 src/ tests/ --max-line-length=100
```

### Code Organization

**Function Structure**:
```python
def function_name(param1: type1, param2: type2 = default) -> return_type:
    """Brief one-line description.
    
    Detailed description if needed, explaining behavior, edge cases, etc.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: value)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is invalid
    """
    # Validate inputs
    if param1 < 0:
        raise ValueError(f"param1 must be ≥ 0, got {param1}")
    
    # Implementation
    result = ...
    
    return result
```

**Error Handling**:
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Recovery or re-raise
    raise
```

### Configuration

**Always use config file, never hardcode**:
```python
# ❌ Bad
hour_bins = 6

# ✅ Good
hour_bins = config['hour_bins']

# ❌ Bad  
k_grid = config.get('k_grid', [1,2,3,4])

# ✅ Good (fail fast)
k_grid = config['k_selection']['k_grid']
```

### Testing

**Write tests for**:
- All new functions
- Bug fixes (regression tests)
- Edge cases and error conditions

**Test Structure**:
```python
def test_feature_behavior():
    """Test that feature behaves correctly under normal conditions."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert np.allclose(result, expected_output, rtol=1e-5)

def test_feature_error_handling():
    """Test that feature raises appropriate errors."""
    with pytest.raises(ValueError, match="param1 must be positive"):
        function_under_test(param1=-1)
```

## Documentation Standards

### Docstring Format (NumPy Style)

```python
def compute_transfer_entropy(source, target, k, tau, num_surrogates=1000):
    """Compute transfer entropy from source to target.
    
    Transfer entropy measures directed information flow using JIDT's
    discrete TE calculator with permutation surrogates for significance
    testing.
    
    Parameters
    ----------
    source : np.ndarray, shape (n_samples,)
        Source time series (discretized integers)
    target : np.ndarray, shape (n_samples,)
        Target time series (discretized integers)
    k : int
        History length
    tau : int
        Time delay
    num_surrogates : int, optional
        Number of permutation surrogates (default: 1000)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'te_value': float, TE in bits
        - 'p_value': float, permutation p-value
        - 'n_samples': int, number of samples used
    
    Raises
    ------
    ValueError
        If k < 1 or tau < 1
        If source and target have different lengths
    RuntimeError
        If JIDT computation fails
    
    Examples
    --------
    >>> source = np.array([0, 1, 1, 0, 1])
    >>> target = np.array([1, 0, 1, 1, 0])
    >>> result = compute_transfer_entropy(source, target, k=1, tau=1)
    >>> result['te_value']
    0.123456
    
    Notes
    -----
    Uses JIDT TransferEntropyCalculatorDiscrete with random permutation
    surrogate generation. See [1]_ for algorithm details.
    
    References
    ----------
    .. [1] Lizier, J. T. (2014). JIDT: An information-theoretic toolkit for
       studying the dynamics of complex systems. Frontiers in Robotics and AI.
    """
```

### README Updates

When adding new features, update:
1. **README.md**: Quick Start, Configuration, Methodology sections
2. **config/README.md**: Configuration options and examples
3. **CHANGELOG.md**: Version history entry

## Commit Messages

**Format**: `<type>(<scope>): <subject>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no functionality change)
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Maintenance (dependencies, build, etc.)

**Examples**:
```
feat(k_selection): add undersampling guard to AIS
fix(cte): correct hour_bins parameter passing
docs(readme): update installation instructions
refactor(preprocessing): extract feature engineering to separate function
```

**Body** (optional, for complex changes):
```
feat(k_selection): add undersampling guard to AIS

- Implement min 25 samples/state constraint
- Add k_max hard cap option
- Track capped users in k_selected output
- Update config schema with undersampling_guard flag

Closes #123
```

## Release Process

1. Update version in `__version__.py`
2. Update CHANGELOG.md with release notes
3. Create PR to `main` with version bump
4. After merge, create GitHub release with tag `vX.Y.Z`
5. Publish to PyPI (if applicable)

## Questions?

- **General questions**: Open a [Discussion](https://github.com/yourusername/extrasensory_analysis/discussions)
- **Bug reports**: Open an [Issue](https://github.com/yourusername/extrasensory_analysis/issues)
- **Security issues**: Email maintainers directly (do not open public issue)

Thank you for contributing! 🎉
