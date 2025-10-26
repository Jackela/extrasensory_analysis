#!/usr/bin/env python
"""Schema validator for ExtraSensory analysis outputs.

Validates per_user_* CSV files against contract v1.0 specification.
"""
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Schema contract v1.0
SCHEMAS = {
    'per_user_te.csv': {
        'required': ['user_id', 'feature_mode', 'k', 'l', 'tau', 'TE_A2S', 'TE_S2A', 
                     'Delta_TE', 'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A'],
        'optional': ['p_Delta_TE', 'q_Delta_TE', 'n_samples', 'low_n'],
        'types': {'k': int, 'l': int, 'tau': int, 'TE_A2S': float, 'TE_S2A': float, 'Delta_TE': float}
    },
    'per_user_cte.csv': {
        'required': ['user_id', 'feature_mode', 'k', 'l', 'tau', 'hour_bins', 
                     'CTE_A2S', 'CTE_S2A', 'Delta_CTE', 'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A'],
        'optional': ['p_Delta_CTE', 'q_Delta_CTE', 'n_samples', 'n_samples_per_bin_min', 'low_n', 'low_n_hours'],
        'types': {'k': int, 'l': int, 'tau': int, 'hour_bins': int, 'CTE_A2S': float, 'Delta_CTE': float},
        'constants': {'hour_bins': 24}
    },
    'per_user_ste.csv': {
        'required': ['user_id', 'feature_mode', 'k', 'tau', 'STE_A2S', 'STE_S2A', 
                     'Delta_STE', 'p_A2S', 'p_S2A', 'q_A2S', 'q_S2A'],
        'optional': ['p_Delta_STE', 'q_Delta_STE'],
        'types': {'k': int, 'tau': int, 'STE_A2S': float, 'STE_S2A': float, 'Delta_STE': float}
    },
    'per_user_gc.csv': {
        'required': ['user_id', 'feature_mode', 'gc_optimal_lag', 'GC_A2S_pval', 'GC_S2A_pval', 
                     'q_GC_A2S', 'q_GC_S2A', 'sign_GC'],
        'optional': [],
        'types': {'gc_optimal_lag': int, 'GC_A2S_pval': float, 'GC_S2A_pval': float}
    }
}


def validate_file(filepath: Path, schema: Dict) -> Tuple[bool, List[str]]:
    """Validate single CSV file against schema.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if not filepath.exists():
        return (False, [f"File not found: {filepath}"])
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return (False, [f"Failed to read CSV: {e}"])
    
    if len(df) == 0:
        errors.append("File is empty")
    
    # Check required columns
    missing = set(schema['required']) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check column types
    for col, expected_type in schema.get('types', {}).items():
        if col in df.columns:
            if expected_type == int:
                if not pd.api.types.is_integer_dtype(df[col]):
                    errors.append(f"Column {col} should be integer type")
            elif expected_type == float:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} should be numeric type")
    
    # Check constant values
    for col, expected_val in schema.get('constants', {}).items():
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0 and not all(v == expected_val for v in unique_vals):
                errors.append(f"Column {col} should be constant {expected_val}, got {unique_vals}")
    
    # Check q columns exist and have values
    q_cols = [c for c in df.columns if c.startswith('q_')]
    if q_cols:
        for q_col in q_cols:
            if df[q_col].notna().sum() == 0:
                errors.append(f"Column {q_col} has no non-null values (FDR not applied?)")
    
    # Check user_id consistency
    if 'user_id' in df.columns:
        if df['user_id'].isna().any():
            errors.append("Column user_id has null values")
    
    return (len(errors) == 0, errors)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ExtraSensory analysis outputs")
    parser.add_argument('--dir', required=True, help="Output directory to validate")
    parser.add_argument('--schema', default='v1.0', help="Schema version")
    args = parser.parse_args()
    
    out_dir = Path(args.dir)
    if not out_dir.exists():
        print(f"ERROR: Directory not found: {out_dir}")
        sys.exit(1)
    
    print(f"Validating outputs in: {out_dir}")
    print(f"Schema version: {args.schema}\n")
    
    all_valid = True
    results = {}
    
    for filename, schema in SCHEMAS.items():
        filepath = out_dir / filename
        is_valid, errors = validate_file(filepath, schema)
        results[filename] = (is_valid, errors)
        
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status} {filename}")
        
        if errors:
            for err in errors:
                print(f"  - {err}")
            all_valid = False
        print()
    
    if all_valid:
        print("=" * 60)
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ VALIDATION FAILED - See errors above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
