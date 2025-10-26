# SSOT Migration: settings.py → YAML Config (2025-10-26)

## Summary
Removed hardcoded configuration constants from `src/settings.py` to enforce Single Source of Truth (SSOT) principle. All configuration now centralized in YAML files.

## Deprecated Constants

### 1. `NUM_SURROGATES` → `config['surrogates']`
**Files affected**: `src/analysis.py` (lines 165, 233), `src/symbolic_te.py` (lines 83, 124, 140)
**Migration**: Pass `surrogates` parameter from config to functions

### 2. `FEATURE_MODES` → `config['feature_modes']`
**Files affected**: `src/preprocessing.py` (line 93)
**Migration**: Pass `feature_modes` list from config for validation

### 3. `EMBEDDING_K_GRID`, `EMBEDDING_TAU_VALUES` → Not in config (unused)
**Files affected**: `src/analysis.py` (lines 325, 329 - robustness_check function)
**Status**: Function never called - can be removed or made config-aware if needed

### 4. `STE_EMBEDDING_DIM`, `STE_DELAY` → Not in config
**Files affected**: `src/symbolic_te.py` (lines 119, 120, 135, 136)
**Migration**: Add to config or keep as STE-specific constants
**Decision**: Keep as algorithm constants (not user-configurable for now)

## Implementation Plan

### Phase 1: Fix Critical Dependencies (NOW)
1. **analysis.py**: Add `num_surrogates` parameter to TE/CTE functions
2. **symbolic_te.py**: Keep STE constants (ordinal_dim=3, delay=1) as defaults
3. **preprocessing.py**: Remove FEATURE_MODES validation (already done via config)

### Phase 2: Update run_production.py
Pass `config['surrogates']` to TE/CTE analysis functions

### Phase 3: Clean settings.py
Keep only:
- Dataset schema constants (COL_*)
- MIN_SAMPLES_PER_BIN (dataset-specific threshold)
- Path constants (JIDT_JAR_PATH, DATA_PATH)

## Breaking Changes

### For Direct Library Users
```python
# OLD
from src import analysis
analysis.run_te_analysis(A, S, k, l, base_A, base_S, tau=1)  # Used settings.NUM_SURROGATES

# NEW
analysis.run_te_analysis(A, S, k, l, base_A, base_S, tau=1, num_surrogates=1000)
```

### For Config Files
Must include:
```yaml
surrogates: 1000  # Explicit, no default
feature_modes: ["composite", "sma_only", "variance_only", "magnitude_only"]
```

## Validation
- [x] Deprecated constants in settings.py
- [ ] Update analysis.py to accept num_surrogates parameter
- [ ] Update run_production.py to pass config['surrogates']
- [ ] Update preprocessing.py validation
- [ ] Test with smoke preset
