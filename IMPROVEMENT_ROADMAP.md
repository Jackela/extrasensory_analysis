# Improvement Roadmap

**Based on**: Architecture Review (2025-10-26)  
**Current Grade**: A- (92/100)  
**Target Grade**: A+ (98/100)

---

## Overview

This roadmap addresses the gaps identified in the architecture review to achieve publication-quality software with comprehensive test coverage and excellent maintainability.

**Total Estimated Effort**: 2 weeks  
**Priority Focus**: Testing infrastructure and code quality

---

## Phase 1: Testing Infrastructure (1 week)

**Goal**: Achieve 80%+ test coverage for critical modules

### Task 1.1: Add `test_preprocessing.py`

**Priority**: HIGH  
**Effort**: 1 day  
**Impact**: Validates data transformation correctness

**Test Cases**:
```python
# test_preprocessing.py

def test_load_subject_data():
    """Test CSV loading for valid user."""
    df = preprocessing.load_subject_data('valid_uuid', 'data_root')
    assert not df.empty
    assert 'timestamp' in df.columns

def test_load_subject_data_missing():
    """Test graceful handling of missing user."""
    df = preprocessing.load_subject_data('invalid_uuid', 'data_root')
    assert df is None

def test_compute_sma():
    """Test simple moving average computation."""
    data = np.array([1, 2, 3, 4, 5])
    sma = preprocessing.compute_sma(data, window=3)
    expected = np.array([np.nan, np.nan, 2, 3, 4])
    np.testing.assert_array_almost_equal(sma, expected, decimal=5)

def test_compute_triaxis_variance():
    """Test triaxial variance calculation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])
    var = preprocessing.compute_triaxis_variance(x, y, z, window=2)
    assert len(var) == len(x)
    assert not np.isnan(var[-1])

def test_create_variables_quintiles():
    """Test quintile discretization."""
    df_in = pd.DataFrame({
        'accel_magnitude': np.random.randn(1000),
        'sitting_probability': np.random.uniform(0, 1, 1000)
    })
    A, S, _ = preprocessing.create_variables(
        df_in, 
        feature_mode='composite',
        hour_bins=6
    )
    # Check quintile discretization (0-4)
    assert A.min() >= 0 and A.max() <= 4
    assert S.min() >= 0 and S.max() <= 4
    assert len(A) == len(S)

def test_create_variables_missing_columns():
    """Test error handling for missing columns."""
    df_in = pd.DataFrame({'invalid_col': [1, 2, 3]})
    with pytest.raises(KeyError):
        preprocessing.create_variables(df_in, 'composite', 6)

def test_zscore_normalization():
    """Test Z-score normalization before discretization."""
    # Input: mean=10, std=2
    data = np.array([8, 10, 12])  # -1σ, 0, +1σ
    normalized = (data - data.mean()) / data.std()
    expected = np.array([-1, 0, 1])
    np.testing.assert_array_almost_equal(normalized, expected, decimal=5)
```

**Validation**:
```bash
pytest tests/test_preprocessing.py -v --cov=src.preprocessing
```

**Target Coverage**: >90% for `preprocessing.py`

---

### Task 1.2: Add `test_quality_control.py`

**Priority**: HIGH  
**Effort**: 2 days  
**Impact**: Validates quality threshold enforcement

**Test Cases**:
```python
# test_quality_control.py

def test_load_quality_profile():
    """Test quality profile loading from YAML."""
    config = yaml.safe_load(open('config/quality/balanced.yaml'))
    qc = QualityController(config)
    assert qc.min_total_samples == 100
    assert qc.te.min_samples == 100
    assert qc.te.min_samples_per_state == 5

def test_validate_global_pass():
    """Test global validation - passing case."""
    qc = QualityController({'quality_control': {
        'global': {'min_total_samples': 100, 'action': 'warn'}
    }})
    result = qc.validate_global(150, 'test_user')
    assert result is True

def test_validate_global_fail_skip():
    """Test global validation - skip action."""
    qc = QualityController({'quality_control': {
        'global': {'min_total_samples': 100, 'action': 'skip'}
    }})
    with pytest.raises(DataQualityError):
        qc.validate_global(50, 'test_user')

def test_validate_te_state_space():
    """Test TE validation with state space check."""
    qc = QualityController({'quality_control': {
        'te': {
            'min_samples': 100,
            'min_samples_per_state': 10,
            'action': 'warn'
        }
    }})
    # State space = 5^4 * 5^5 = 625 * 3125 = 1,953,125
    # Need 1,953,125 * 10 = 19,531,250 samples
    result = qc.validate_te(
        n_samples=1000, 
        user_id='test', 
        base_A=5, 
        base_S=5, 
        k=4
    )
    assert result is False  # Should warn, not pass

def test_validate_cte_bin_filtering():
    """Test CTE bin filtering logic."""
    qc = QualityController({'quality_control': {
        'cte': {
            'min_total_samples': 200,
            'min_bin_samples': 50,
            'max_filtered_bins': 2,
            'max_filtered_ratio': 0.5,
            'action': 'filter_bins'
        }
    }})
    hour_counts = [100, 80, 60, 45, 30, 20]  # 2 bins below threshold
    passed, valid_bins, diag = qc.validate_cte(500, hour_counts, 'test')
    
    assert passed is True
    assert len(valid_bins) == 4  # First 4 bins pass
    assert diag['filtered_count'] == 2
    assert 4 in diag['low_bins']
    assert 5 in diag['low_bins']

def test_estimate_min_samples_te():
    """Test dynamic TE sample estimation."""
    qc = QualityController({'quality_control': {
        'te': {'min_samples_per_state': 10}
    }})
    # k=4: state_space = 5^4 * 5^5 = 1,953,125
    min_samples = qc.estimate_min_samples_te(base_A=5, base_S=5, k=4)
    assert min_samples == 1953125 * 10

def test_recommend_max_k():
    """Test k recommendation based on sample size."""
    qc = QualityController({'quality_control': {
        'k_selection': {
            'min_samples_per_state': 25,
            'max_k_absolute': 6
        }
    }})
    # 5000 samples / 25 sps = 200 states max
    # 5^k <= 200 → k <= 3 (5^3 = 125)
    rec_k = qc.recommend_max_k(n_samples=5000, base=5)
    assert rec_k == 3

def test_estimate_statistical_power():
    """Test statistical power estimation."""
    qc = QualityController({})
    
    # Low power: <1 sample/state
    assert qc.estimate_statistical_power(500, 1000) == 0.20
    
    # Medium power: ~5 samples/state
    assert qc.estimate_statistical_power(5000, 1000) == 0.70
    
    # High power: >10 samples/state
    assert qc.estimate_statistical_power(10000, 1000) == 0.80

def test_quality_diagnostics():
    """Test comprehensive quality diagnostics."""
    qc = QualityController({'quality_control': {
        'global': {'min_total_samples': 100},
        'te': {'min_samples': 100, 'min_samples_per_state': 5},
        'cte': {'min_total_samples': 200}
    }})
    
    diag = qc.get_quality_diagnostics(
        n_samples=2287,
        base_A=5,
        base_S=5,
        k=4
    )
    
    assert diag['n_samples'] == 2287
    assert diag['global_passed'] is True
    assert 'te' in diag
    assert 'k_selection' in diag
    assert len(diag['recommendations']) > 0

def test_generate_quality_summary():
    """Test quality summary generation."""
    qc = QualityController({'quality_control': {
        'global': {'min_total_samples': 100}
    }})
    
    diag = qc.get_quality_diagnostics(n_samples=2287, base_A=5, base_S=5, k=4)
    summary = qc.generate_quality_summary(diag)
    
    assert 'Data Quality Summary' in summary
    assert '2,287' in summary
    assert 'Global:' in summary
```

**Validation**:
```bash
pytest tests/test_quality_control.py -v --cov=src.quality_control
```

**Target Coverage**: >85% for `quality_control.py`

---

### Task 1.3: Add `test_fdr_utils.py`

**Priority**: HIGH  
**Effort**: 1 day  
**Impact**: Validates statistical correctness

**Test Cases**:
```python
# test_fdr_utils.py

def test_apply_fdr_single_family():
    """Test BH correction on single family."""
    df = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3'],
        'p_A2S': [0.01, 0.05, 0.10],
        'p_S2A': [0.02, 0.08, 0.15]
    })
    
    df_corrected = apply_fdr_per_family_tau(
        df, 
        p_cols=['p_A2S', 'p_S2A'],
        q_cols=['q_A2S', 'q_S2A'],
        family='TE',
        tau_col=None,
        alpha=0.05
    )
    
    # BH correction should increase q-values
    assert all(df_corrected['q_A2S'] >= df_corrected['p_A2S'])
    assert all(df_corrected['q_S2A'] >= df_corrected['p_S2A'])
    
    # Smallest p-value should remain significant
    assert df_corrected.loc[0, 'q_A2S'] < 0.05

def test_apply_fdr_per_tau():
    """Test FDR correction stratified by tau."""
    df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2', 'u2'],
        'tau': [1, 2, 1, 2],
        'p_A2S': [0.01, 0.02, 0.03, 0.04],
        'p_S2A': [0.05, 0.06, 0.07, 0.08]
    })
    
    df_corrected = apply_fdr_per_family_tau(
        df,
        p_cols=['p_A2S', 'p_S2A'],
        q_cols=['q_A2S', 'q_S2A'],
        family='TE',
        tau_values=[1, 2],
        alpha=0.05
    )
    
    # Should have separate corrections per tau
    tau1_q = df_corrected[df_corrected['tau'] == 1]['q_A2S'].values
    tau2_q = df_corrected[df_corrected['tau'] == 2]['q_A2S'].values
    
    # Within each tau group, BH correction applied
    assert len(tau1_q) == 2
    assert len(tau2_q) == 2

def test_compute_delta_pvalue():
    """Test Delta (A→S - S→A) p-value computation."""
    df = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3'],
        'Delta_TE': [0.05, -0.02, 0.10]
    })
    
    sign, p_delta = compute_delta_pvalue(df, 'Delta_TE', 'p_Delta')
    
    # Wilcoxon signed-rank test
    assert isinstance(sign, str)
    assert sign in ['A2S', 'S2A', 'ns']
    assert 'p_Delta' in df.columns
    assert 0 <= df['p_Delta'].iloc[0] <= 1

def test_fdr_correction_strictness():
    """Test that FDR is more conservative than uncorrected."""
    df = pd.DataFrame({
        'p_A2S': [0.001, 0.01, 0.05, 0.10, 0.20]
    })
    
    # Apply BH correction manually
    from statsmodels.stats.multitest import multipletests
    reject, q_vals, _, _ = multipletests(df['p_A2S'], alpha=0.05, method='fdr_bh')
    
    # q-values should be >= p-values
    assert all(q_vals >= df['p_A2S'])
    
    # Fewer rejections with FDR
    assert sum(reject) <= sum(df['p_A2S'] < 0.05)
```

**Validation**:
```bash
pytest tests/test_fdr_utils.py -v --cov=src.fdr_utils
```

**Target Coverage**: >95% for `fdr_utils.py`

---

### Task 1.4: Add Integration Tests

**Priority**: HIGH  
**Effort**: 1 day  
**Impact**: Validates end-to-end system behavior

**Test Cases**:
```python
# test_integration.py

def test_smoke_preset_validation():
    """Test smoke preset configuration loading."""
    with open('config/presets/smoke.yaml') as f:
        config = yaml.safe_load(f)
    
    # Required fields present
    assert 'data_root' in config
    assert 'out_dir' in config
    assert 'n_users' in config
    assert 'feature_modes' in config
    assert 'taus' in config
    assert 'surrogates' in config
    assert 'quality_profile' in config
    
    # Smoke test constraints
    assert config['n_users'] == 2
    assert config['surrogates'] == 100

def test_quality_profile_validation():
    """Test all quality profiles load correctly."""
    profiles = ['strict', 'balanced', 'exploratory']
    
    for profile in profiles:
        path = f'config/quality/{profile}.yaml'
        with open(path) as f:
            config = yaml.safe_load(f)
        
        qc = QualityController(config)
        
        # Required thresholds present
        assert qc.min_total_samples > 0
        assert qc.te.min_samples > 0
        assert qc.cte.min_total_samples > 0
        assert qc.k_selection.min_samples_per_state > 0

def test_pipeline_smoke_run():
    """Test end-to-end pipeline execution."""
    # This would require actual data, skip if unavailable
    pytest.importorskip('data')
    
    from run_production import ProductionPipeline
    
    pipeline = ProductionPipeline('config/presets/smoke.yaml')
    
    # Should initialize without errors
    assert pipeline.config['n_users'] == 2
    assert pipeline.quality is not None
    
    # Mock user list
    user_list = ['test_user_1', 'test_user_2']
    
    # Should create output directory
    assert pipeline.out_dir.exists()

def test_csv_output_schema():
    """Test that CSV outputs match expected schema."""
    # Load sample output (from smoke test)
    te_path = 'analysis/out/smoke_*/per_user_te.csv'
    
    import glob
    te_files = glob.glob(te_path)
    if not te_files:
        pytest.skip("No smoke test outputs available")
    
    df = pd.read_csv(te_files[0])
    
    # Expected columns
    expected_cols = [
        'user_id', 'feature_mode', 'k', 'l', 'tau',
        'TE_A2S', 'TE_S2A', 'Delta_TE',
        'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A',
        'p_Delta_TE', 'q_Delta_TE',
        'n_samples', 'low_n', 'quality_passed'
    ]
    
    for col in expected_cols:
        assert col in df.columns
    
    # Data types
    assert df['k'].dtype in [np.int32, np.int64]
    assert df['TE_A2S'].dtype in [np.float32, np.float64]
    assert df['quality_passed'].dtype == bool

def test_quality_report_generation():
    """Test quality report generation."""
    import glob
    report_files = glob.glob('analysis/out/smoke_*/quality_report.md')
    
    if not report_files:
        pytest.skip("No quality reports available")
    
    with open(report_files[0]) as f:
        content = f.read()
    
    # Required sections
    assert '# Quality Control Report' in content
    assert '## Summary Statistics' in content
    assert '### Transfer Entropy (TE)' in content
    assert '### Conditional Transfer Entropy (CTE)' in content
    assert '## Quality Violations' in content
```

**Validation**:
```bash
pytest tests/test_integration.py -v
```

---

### Task 1.5: Coverage Tracking Setup

**Priority**: MEDIUM  
**Effort**: 2 hours  
**Impact**: Visibility into test quality

**Steps**:

1. **Install pytest-cov**:
   ```bash
   pip install pytest-cov
   echo "pytest-cov>=4.0.0" >> requirements.txt
   ```

2. **Configure pytest.ini**:
   ```ini
   # pytest.ini
   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   
   addopts =
       --verbose
       --cov=src
       --cov=run_production
       --cov-report=term-missing
       --cov-report=html:htmlcov
       --cov-fail-under=80
   ```

3. **Add .coveragerc**:
   ```ini
   # .coveragerc
   [run]
   source = src,run_production
   omit =
       tests/*
       */site-packages/*
   
   [report]
   exclude_lines =
       pragma: no cover
       def __repr__
       raise AssertionError
       raise NotImplementedError
       if __name__ == .__main__.:
   ```

4. **Run with coverage**:
   ```bash
   pytest --cov=src --cov-report=term-missing --cov-report=html
   ```

5. **Add to CI/CD** (if applicable):
   ```yaml
   # .github/workflows/tests.yml
   - name: Run tests with coverage
     run: |
       pytest --cov=src --cov-report=xml
   
   - name: Upload coverage to Codecov
     uses: codecov/codecov-action@v3
   ```

**Target**: Achieve 80%+ overall coverage

---

## Phase 2: Code Quality Improvements (2 days)

**Goal**: Improve readability and maintainability

### Task 2.1: Extract Helper Functions

**Priority**: MEDIUM  
**Effort**: 4 hours  
**Impact**: Improves code readability

**Locations**:

1. **`analysis.run_cte_analysis()` (74 lines)**:
   ```python
   # Extract stratification logic
   def _stratify_by_hour_bins(A, S, H_binned, num_bins):
       """Extract time series per hour bin."""
       stratified_data = {}
       for h in range(num_bins):
           mask = (H_binned == h)
           stratified_data[h] = {
               'A': A[mask],
               'S': S[mask],
               'n': mask.sum()
           }
       return stratified_data
   
   # Extract aggregation logic
   def _aggregate_cte_results(bin_results):
       """Aggregate CTE across bins using Fisher's method."""
       # ... Fisher combination logic
       return aggregated_cte
   ```

2. **`quality_control.generate_quality_report()` (103 lines)**:
   ```python
   # Extract formatters
   def _format_method_summary(df, method_name):
       """Generate summary statistics for analysis method."""
       total = len(df)
       passed = df['quality_passed'].sum()
       mean_samples = df['n_samples'].mean()
       return f"- Total: {total}\n- Passed: {passed}\n- Mean samples: {mean_samples:.0f}"
   
   def _format_violations(results):
       """Format quality violation summary."""
       violations = []
       for method, df in results.items():
           failed = df[~df['quality_passed']]
           if len(failed) > 0:
               violations.append(f"- {method}: {len(failed)} failed")
       return violations
   ```

**Validation**:
- Run existing tests to ensure no regressions
- Check that extracted functions are reusable

---

### Task 2.2: Add Type Hints

**Priority**: MEDIUM  
**Effort**: 4 hours  
**Impact**: Improves type safety

**Approach**:

1. **Add type hints to all function signatures**:
   ```python
   # Before
   def load_subject_data(user_id, data_root):
       ...
   
   # After
   from typing import Optional
   import pandas as pd
   
   def load_subject_data(user_id: str, data_root: str) -> Optional[pd.DataFrame]:
       """Load subject data from CSV."""
       ...
   ```

2. **Add return type annotations**:
   ```python
   def compute_sma(data: np.ndarray, window: int) -> np.ndarray:
       """Compute simple moving average."""
       return pd.Series(data).rolling(window).mean().values
   ```

3. **Use typing generics for complex types**:
   ```python
   from typing import Dict, List, Tuple
   
   def validate_cte(
       self, 
       n_samples: int, 
       hour_counts: List[int],
       user_id: str
   ) -> Tuple[bool, List[int], Dict[str, Any]]:
       """Validate CTE requirements."""
       ...
   ```

4. **Run mypy validation**:
   ```bash
   pip install mypy
   mypy src/ --ignore-missing-imports
   ```

**Files to update**:
- All modules in `src/`
- `run_production.py`
- `tools/` scripts

---

### Task 2.3: Code Review & Cleanup

**Priority**: MEDIUM  
**Effort**: 4 hours  
**Impact**: Identifies technical debt

**Checklist**:

- [ ] Review all 70+ line functions
- [ ] Check for duplicated code
- [ ] Validate error handling patterns
- [ ] Review logging consistency
- [ ] Check for magic numbers
- [ ] Validate docstring completeness
- [ ] Review import organization
- [ ] Check for unused imports/variables

**Tools**:
```bash
# Find long functions
grep -n "^def " src/*.py | while read line; do
  file=$(echo $line | cut -d: -f1)
  func=$(echo $line | cut -d: -f3)
  lines=$(wc -l < "$file")
  echo "$file:$func ($lines lines)"
done

# Find duplicated code
pip install pylint
pylint src/ --disable=all --enable=duplicate-code

# Find unused imports
pip install autoflake
autoflake --remove-all-unused-imports --check src/
```

---

## Phase 3: Documentation (3 days)

**Goal**: Complete documentation coverage

### Task 3.1: Generate API Documentation

**Priority**: LOW  
**Effort**: 2 days  
**Impact**: Improves developer experience

**Steps**:

1. **Install Sphinx**:
   ```bash
   pip install sphinx sphinx-rtd-theme
   ```

2. **Initialize Sphinx**:
   ```bash
   cd docs/
   sphinx-quickstart
   ```

3. **Configure Sphinx** (`docs/conf.py`):
   ```python
   import os
   import sys
   sys.path.insert(0, os.path.abspath('..'))
   
   project = 'ExtraSensory Analysis'
   author = 'Research Team'
   
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.napoleon',
       'sphinx.ext.viewcode',
   ]
   
   templates_path = ['_templates']
   exclude_patterns = []
   
   html_theme = 'sphinx_rtd_theme'
   ```

4. **Generate documentation**:
   ```bash
   sphinx-apidoc -f -o docs/source src/
   cd docs && make html
   ```

5. **Add usage examples to docstrings**:
   ```python
   def load_subject_data(user_id: str, data_root: str) -> Optional[pd.DataFrame]:
       """
       Load subject data from ExtraSensory CSV files.
       
       Args:
           user_id: Subject UUID
           data_root: Path to ExtraSensory dataset
       
       Returns:
           DataFrame with sensor readings or None if file not found
       
       Example:
           >>> df = load_subject_data('00EABED2-...', 'data/')
           >>> print(df.columns)
           ['timestamp', 'accel_x', 'accel_y', ...]
       """
       ...
   ```

---

### Task 3.2: Create Architecture Diagrams

**Priority**: LOW  
**Effort**: 1 day  
**Impact**: Visual understanding

**Diagrams to Create**:

1. **Module Dependency Graph**:
   ```mermaid
   graph TD
       A[settings.py] --> B[preprocessing.py]
       A --> C[analysis.py]
       A --> D[jidt_adapter.py]
       E[params.py] --> C
       E --> D
       B --> F[run_production.py]
       C --> F
       G[quality_control.py] --> F
   ```

2. **Data Flow Diagram**:
   ```mermaid
   graph LR
       A[CSV Files] --> B[load_subject_data]
       B --> C[create_variables]
       C --> D[Z-score + Quintile]
       D --> E[A, S, H arrays]
       E --> F[JIDT Analysis]
       F --> G[FDR Correction]
       G --> H[CSV Outputs]
   ```

3. **Configuration Hierarchy**:
   ```mermaid
   graph TD
       A[config/presets/] --> B[smoke.yaml]
       A --> C[k6_full.yaml]
       A --> D[k4_strict.yaml]
       E[config/quality/] --> F[strict.yaml]
       E --> G[balanced.yaml]
       E --> H[exploratory.yaml]
       B --> I[QualityController]
       F --> I
   ```

**Tools**: Mermaid, draw.io, or PlantUML

---

## Summary

### Effort Breakdown

| Phase | Tasks | Effort | Priority |
|-------|-------|--------|----------|
| Phase 1 | Testing | 1 week | HIGH |
| Phase 2 | Code Quality | 2 days | MEDIUM |
| Phase 3 | Documentation | 3 days | LOW |
| **Total** | | **2 weeks** | |

### Expected Outcomes

**After Phase 1** (1 week):
- ✅ 80%+ test coverage
- ✅ All critical modules tested
- ✅ Integration tests passing
- ✅ Coverage tracking enabled

**After Phase 2** (1.5 weeks):
- ✅ Improved code readability
- ✅ Complete type annotations
- ✅ Technical debt identified
- ✅ Refactoring complete

**After Phase 3** (2 weeks):
- ✅ API documentation generated
- ✅ Architecture diagrams created
- ✅ A+ grade achieved (98/100)

### Success Criteria

- [ ] Test coverage ≥80%
- [ ] All critical modules have unit tests
- [ ] Integration tests pass
- [ ] Type hints on all functions
- [ ] API documentation generated
- [ ] Architecture diagrams complete
- [ ] Code review complete
- [ ] No regressions in existing functionality

---

**Roadmap Owner**: Development Team  
**Review Date**: 2025-11-09 (2 weeks from start)  
**Target Grade**: A+ (98/100)

