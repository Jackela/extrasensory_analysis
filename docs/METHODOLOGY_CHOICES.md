# Methodology Choices Rationale

This document records the reasoning behind key methodological choices made during the ExtraSensory Transfer Entropy analysis project.

## 1. K-Selection Strategy: Active Information Storage (AIS)

**Choice**: Use Active Information Storage (AIS) to automatically select the optimal history length (`k`) for each user's time series (`k_selection.strategy: "AIS"` in configuration presets like `k6_full`).

**Rationale**:
* **Data-Driven**: AIS quantifies how much information the past `k` states of a variable contain about its *next* state ($AIS_k = I(X_t; X_{t-k:t-1})$). Selecting `k` that maximizes AIS (or where AIS plateaus) ensures the chosen history length is empirically optimal for self-prediction based on the data itself, rather than relying on an arbitrary fixed value.
* **Principled**: This approach aligns with information-theoretic best practices for selecting embedding parameters in time series analysis. The project proposal explicitly mentions leveraging JIDT's automated routines based on optimizing information-theoretic criteria like AIS.
* **Avoids Arbitrary Choice**: Using a fixed `k` for all users would ignore individual differences in behavioral dynamics. AIS allows `k` to adapt to each user's data characteristics.
* **Implementation**: The `src/k_selection.py` module implements AIS calculation using JIDT's `ActiveInformationCalculatorDiscrete` and selects `k` based on criteria defined in the configuration files.

## 2. CTE Hour Bins: 6 Bins (4-Hour Intervals)

**Choice**: Use 6 bins for stratifying the hour-of-day (`hour_bins: 6`) when calculating Conditional Transfer Entropy (CTE), corresponding to 4-hour intervals (00-04, 04-08, ..., 20-24).

**Rationale**:
* **Balance**: This choice represents a balance between temporal resolution and data availability per bin.
* **Computational Feasibility & Data Sparsity**: Using 24 bins (1-hour intervals) significantly increases the risk of data sparsity within each bin. Reliable TE estimation requires sufficient samples per bin. Testing showed that even with a `min_bin_samples` threshold of 100-150, using 6 bins resulted in minimal data filtering across the 60 users, suggesting adequate data density within 4-hour windows for most participants. Using 24 bins would likely lead to much more filtering or require lower (less reliable) sample thresholds.
* **Circadian Relevance**: 4-hour blocks (e.g., morning, midday, afternoon, evening, late night, early morning) capture meaningful segments of daily human activity patterns and circadian rhythms, while still aggregating enough data within each block.
* **Default in Presets**: This is the default setting in primary analysis presets like `k6_full` and `k4_fast`. A high-resolution `24bin_cte` preset exists but requires careful consideration of data sparsity and typically lower `k` values.

