# Configuration Migration Notes (2025-10-26)

## Summary
Removed hardcoded configuration constants from `src/settings.py` and migrated to centralized YAML-based configuration.

## Changes Made

### 1. Removed Hardcoded Constants
**File**: `src/settings.py`
- **Removed**: `NUM_HOUR_BINS = 6` (line 28)
- **Replaced with**: Config parameter `hour_bins` in `config/template.yaml (reference) or config/presets/`
- **Migration**: All code now reads `hour_bins` from config, not hardcoded constant

### 2. Updated Function Signatures
**File**: `src/preprocessing.py`
- **Before**: `create_variables(df, feature_mode='composite')`
- **After**: `create_variables(df, feature_mode='composite', hour_bins=None)`
- **Validation**: Raises error if `hour_bins` is not passed from config

### 3. Added Config Validation
**File**: `run_production.py`
- Added schema validation in `__init__()`:
  - Required fields check: `hour_bins`, `taus`, `k_selection`, etc.
  - Type validation: `hour_bins` must be int ≥ 1
  - Fail-fast behavior: Errors raised before processing starts

### 4. Cleaned Project Structure
**Deprecated files moved to** `archive/deprecated_20251026/`:
- `run_ais_guarded.py` → Integrated into run_production.py
- `run_ais_scan.py` → One-time tool, no longer needed
- `run_proposal_pipeline.py` → Replaced by run_production.py
- `run_smoke_corrected.py` → Replaced by run_production.py --smoke
- `FINAL_COMPLIANCE_REPORT.md` → Process documentation
- `.cleanup_plan.txt` → Temporary planning file

## Configuration Best Practices

1. **Explicit Configuration**: No default values in function signatures
2. **Centralized Source**: Single YAML file for all parameters
3. **Schema Validation**: Validate on startup, fail fast
4. **Type Safety**: Enforce type constraints (int, list, dict)
5. **No Hidden State**: All behavior controlled via config

## Breaking Changes

### For Library Users
If you were using `preprocessing.create_variables()` directly:
```python
# OLD (will now fail)
A, S, H_raw, H_binned = create_variables(df, feature_mode='composite')

# NEW (required)
A, S, H_raw, H_binned = create_variables(df, feature_mode='composite', hour_bins=6)
```

### For Config Files
All custom config files must now include:
```yaml
hour_bins: 6  # Required, no default
k_selection:
  strategy: "AIS"  # or "GUARDED_AIS" or "FIXED"
  k_grid: [1, 2, 3, 4, 5, 6]
```

## Backward Compatibility

**Not Maintained**: This is a breaking change. Old code using hardcoded constants will fail with clear error messages directing users to update their configuration.

## Testing Checklist

- [x] Config validation passes
- [x] Smoke test with 6-bin CTE runs successfully
- [x] Missing config fields raise clear errors
- [x] Invalid hour_bins values rejected
- [ ] Full 60-user run pending user approval

## Next Steps

1. Run full 60-user × 4-mode analysis with validated config
2. Monitor for any config-related issues
3. Update any external scripts/notebooks that call preprocessing functions
