# Quality Control Configuration

Quality threshold profiles for statistical rigor and data quality management.

## Available Profiles

### 🔴 **Strict** (`strict.yaml`)
**Use for**: Publication-quality results, hypothesis testing, clinical applications

- **Global**: min 200 samples, skip users below threshold
- **TE**: min 200 samples, 10 samples/state
- **CTE**: min 500 samples, 100/bin, max 2 filtered bins
- **k-selection**: 50 samples/state, max k=4
- **Action**: Skip low-quality users entirely

**Trade-off**: Higher statistical power but reduced sample size (expect ~60-70% user retention)

---

### 🟡 **Balanced** (`balanced.yaml`) - **RECOMMENDED**
**Use for**: Most research applications, production pipelines

- **Global**: min 100 samples, warn on violations
- **TE**: min 100 samples, 5 samples/state
- **CTE**: min 200 samples, 50/bin, max 4 filtered bins
- **k-selection**: 25 samples/state, max k=6
- **Action**: Warn and flag, filter only severe cases

**Trade-off**: Good balance between rigor and data retention (~85-90% user retention)

---

### 🟢 **Exploratory** (`exploratory.yaml`)
**Use for**: Pattern discovery, hypothesis generation, feasibility studies

- **Global**: min 50 samples, warn only
- **TE**: min 50 samples, 3 samples/state
- **CTE**: min 100 samples, 30/bin, no filtering
- **k-selection**: 15 samples/state, no enforcement
- **Action**: Warn only, never skip

**Trade-off**: Maximum data retention but lower statistical confidence (~95-100% retention)

---

## Scientific Rationale

### Sample Size Requirements

#### Transfer Entropy (TE)
- **State space**: `base_dest^k_dest × base_src^(k_src+1)`
- **Example** (k=4, base_A=5, base_S=2): 625 × 32 = 20,000 states
- **Theoretical**: 25-50 samples/state → 500K-1M samples (unrealistic)
- **Practical compromise**: 
  - Strict: 10 samples/state → 200K states × 10 = 200+ samples minimum
  - Balanced: 5 samples/state → 100+ samples
  - Exploratory: 3 samples/state → 50+ samples

#### Conditional Transfer Entropy (CTE)
- **Additional complexity**: Hour bin stratification reduces effective N
- **Rule of thumb**: 2-3× TE requirements due to data splitting
- **Bin filtering**: Remove bins with <50 samples (balanced) to prevent bias

#### Symbolic Transfer Entropy (STE)
- **Ordinal patterns**: 3! = 6 possible patterns per dimension
- **More robust**: Ordinal encoding reduces sensitivity to outliers
- **Lower requirements**: Can work with ~50% of TE requirements

#### Active Information Storage (AIS) for k-selection
- **State space**: `base^k`
- **Example** (base=2, k=6): 2^6 = 64 states
- **Undersampling guard**: Prevent k where samples/state < 25
- **Protection**: Avoid overfitting in high-dimensional spaces

---

## References

1. **Lizier, J. T. (2014)**. "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems." *Frontiers in Robotics and AI*, 1, 11.
   - Establishes baseline sample requirements for discrete TE estimation

2. **Wibral, M., Vicente, R., & Lizier, J. T. (2014)**. "Directed information measures in neuroscience." *Springer*.
   - Chapter on sample size and bias in TE estimation

3. **Schreiber, T. (2000)**. "Measuring information transfer." *Physical Review Letters*, 85(2), 461.
   - Original TE paper, discusses estimation reliability

4. **Kraskov, A., Stögbauer, H., & Grassberger, P. (2004)**. "Estimating mutual information." *Physical Review E*, 69(6), 066138.
   - k-NN estimation methods and sample requirements

5. **Wollstadt, P., et al. (2019)**. "IDTxl: Information Dynamics Toolkit xl." *JOSS*, 4(34), 1081.
   - Modern best practices for information-theoretic analysis

---

## Usage in Presets

```yaml
# config/presets/your_preset.yaml
quality_profile: "balanced"  # strict | balanced | exploratory
```

Or override specific thresholds:

```yaml
quality_control:
  global:
    min_total_samples: 150  # Custom threshold
  te:
    min_samples: 120
```

---

## Quality Flags in Output

All output CSVs include quality indicators:

- `low_n`: Boolean flag for total sample count
- `low_n_hours`: JSON array of hour bins below threshold (CTE only)
- `n_samples`: Actual sample count for verification
- `quality_passed`: Overall quality assessment (future)

**Recommendation**: Filter results by `low_n == False` for high-confidence analyses.

---

## Migration from Legacy Thresholds

### Old (hardcoded)
```python
MIN_SAMPLES_PER_BIN = 30  # settings.py
low_n = len(A) < 100      # run_production.py
if n_h < 50: skip         # jidt_adapter.py
```

### New (configurable)
```yaml
quality_control:
  cte:
    min_bin_samples: 50
  te:
    min_samples: 100
```

All thresholds now centralized and scientifically justified.
