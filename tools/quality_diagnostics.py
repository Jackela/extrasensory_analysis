#!/usr/bin/env python
"""
Quality diagnostics utility for ExtraSensory pipeline.

Demonstrates dynamic threshold estimation and statistical power calculation.
"""

import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.quality_control import QualityController


def demonstrate_diagnostics():
    """Demonstrate quality diagnostic capabilities."""
    
    print("=" * 70)
    print("ExtraSensory Quality Control - Diagnostic Utilities")
    print("=" * 70)
    
    # Load balanced profile
    profile_path = Path('config/quality/balanced.yaml')
    with open(profile_path) as f:
        config = yaml.safe_load(f)
    
    qc = QualityController(config)
    
    # Example 1: Small dataset (2287 samples from smoke test)
    print("\n### Example 1: Small Dataset (n=2287)")
    print("-" * 70)
    
    diagnostics = qc.get_quality_diagnostics(
        n_samples=2287,
        base_A=5,  # 5-level discretization
        base_S=5,
        k=4
    )
    
    summary = qc.generate_quality_summary(diagnostics)
    print(summary)
    
    # Example 2: Medium dataset (6808 samples from smoke test)
    print("\n\n### Example 2: Medium Dataset (n=6808)")
    print("-" * 70)
    
    diagnostics = qc.get_quality_diagnostics(
        n_samples=6808,
        base_A=5,
        base_S=5,
        k=4
    )
    
    summary = qc.generate_quality_summary(diagnostics)
    print(summary)
    
    # Example 3: k-selection recommendations
    print("\n\n### Example 3: k-Selection Recommendations")
    print("-" * 70)
    
    sample_sizes = [1000, 5000, 10000, 20000, 50000]
    base = 5
    
    print(f"{'n_samples':>10} | {'Recommended k':>15} | {'Min samples@k':>20}")
    print("-" * 70)
    
    for n in sample_sizes:
        rec_k = qc.recommend_max_k(n, base)
        min_samples = qc.estimate_min_samples_ais(base, rec_k)
        print(f"{n:>10,} | {rec_k:>15} | {min_samples:>20,}")
    
    # Example 4: Statistical power estimation
    print("\n\n### Example 4: Statistical Power vs Sample Size")
    print("-" * 70)
    
    k = 4
    state_space = 5 ** k  # 625 states
    
    print(f"State space size: {state_space:,} (base={5}, k={k})")
    print(f"\n{'n_samples':>10} | {'Samples/state':>15} | {'Est. Power':>12} | {'Level':>8}")
    print("-" * 70)
    
    for n in [500, 1000, 2000, 5000, 10000]:
        sps = n / state_space
        power = qc.estimate_statistical_power(n, state_space)
        level = 'high' if power >= 0.80 else 'medium' if power >= 0.70 else 'low'
        print(f"{n:>10,} | {sps:>15.2f} | {power:>12.2f} | {level:>8}")
    
    # Example 5: TE minimum sample estimation
    print("\n\n### Example 5: TE Minimum Sample Requirements")
    print("-" * 70)
    
    print(f"{'k':>5} | {'State Space':>15} | {'Min Samples':>15}")
    print("-" * 70)
    
    for k in range(1, 7):
        state_space = (5 ** k) * (5 ** (k + 1))
        min_samples = qc.estimate_min_samples_te(base_A=5, base_S=5, k=k)
        print(f"{k:>5} | {state_space:>15,} | {min_samples:>15,}")
    
    print("\n" + "=" * 70)
    print("Diagnostics complete. Use these utilities for:")
    print("  • Pre-flight sample size validation")
    print("  • k-selection parameter tuning")
    print("  • Statistical power analysis")
    print("  • Quality threshold calibration")
    print("=" * 70 + "\n")


def compare_profiles():
    """Compare quality thresholds across profiles."""
    
    print("\n" + "=" * 70)
    print("Quality Profile Comparison")
    print("=" * 70 + "\n")
    
    profiles = ['strict', 'balanced', 'exploratory']
    configs = {}
    
    for profile in profiles:
        path = Path(f'config/quality/{profile}.yaml')
        with open(path) as f:
            configs[profile] = yaml.safe_load(f)
    
    print(f"{'Threshold':>30} | {'Strict':>10} | {'Balanced':>10} | {'Exploratory':>12}")
    print("-" * 70)
    
    # Global thresholds
    print("GLOBAL:")
    for profile in profiles:
        qc_cfg = configs[profile]['quality_control']
        min_samples = qc_cfg['global']['min_total_samples']
        action = qc_cfg['global']['action']
        print(f"{'  min_samples':>30} | {min_samples if profile == 'strict' else '':>10} | {min_samples if profile == 'balanced' else '':>10} | {min_samples if profile == 'exploratory' else '':>12}")
    
    print(f"{'':>30} | {configs['strict']['quality_control']['global']['min_total_samples']:>10} | {configs['balanced']['quality_control']['global']['min_total_samples']:>10} | {configs['exploratory']['quality_control']['global']['min_total_samples']:>12}")
    
    # TE thresholds
    print("\nTE:")
    print(f"{'  min_samples':>30} | {configs['strict']['quality_control']['te']['min_samples']:>10} | {configs['balanced']['quality_control']['te']['min_samples']:>10} | {configs['exploratory']['quality_control']['te']['min_samples']:>12}")
    print(f"{'  min_samples/state':>30} | {configs['strict']['quality_control']['te']['min_samples_per_state']:>10} | {configs['balanced']['quality_control']['te']['min_samples_per_state']:>10} | {configs['exploratory']['quality_control']['te']['min_samples_per_state']:>12}")
    
    # CTE thresholds
    print("\nCTE:")
    print(f"{'  min_total_samples':>30} | {configs['strict']['quality_control']['cte']['min_total_samples']:>10} | {configs['balanced']['quality_control']['cte']['min_total_samples']:>10} | {configs['exploratory']['quality_control']['cte']['min_total_samples']:>12}")
    print(f"{'  min_bin_samples':>30} | {configs['strict']['quality_control']['cte']['min_bin_samples']:>10} | {configs['balanced']['quality_control']['cte']['min_bin_samples']:>10} | {configs['exploratory']['quality_control']['cte']['min_bin_samples']:>12}")
    
    # k-selection
    print("\nk-SELECTION:")
    print(f"{'  min_samples/state':>30} | {configs['strict']['quality_control']['k_selection']['min_samples_per_state']:>10} | {configs['balanced']['quality_control']['k_selection']['min_samples_per_state']:>10} | {configs['exploratory']['quality_control']['k_selection']['min_samples_per_state']:>12}")
    print(f"{'  max_k':>30} | {configs['strict']['quality_control']['k_selection']['max_k_absolute']:>10} | {configs['balanced']['quality_control']['k_selection']['max_k_absolute']:>10} | {configs['exploratory']['quality_control']['k_selection']['max_k_absolute']:>12}")
    
    # Statistical settings
    print("\nSTATISTICAL:")
    print(f"{'  min_power':>30} | {configs['strict']['statistical']['min_power']:>10} | {configs['balanced']['statistical']['min_power']:>10} | {configs['exploratory']['statistical']['min_power']:>12}")
    
    print("=" * 70 + "\n")


if __name__ == '__main__':
    demonstrate_diagnostics()
    compare_profiles()
