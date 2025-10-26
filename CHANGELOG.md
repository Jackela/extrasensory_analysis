# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pure AIS k-selection with optional k_max and undersampling constraints
- 6-bin hour-of-day CTE stratification (4-hour windows)
- Checkpoint/resume functionality for long-running analyses
- Real-time status tracking (status.json with progress and ETA)
- Comprehensive run metadata (run_info.yaml with JIDT version, git commit, config snapshot)
- Configurable k-selection strategies (AIS, GUARDED_AIS, FIXED)
- FDR correction per (method, tau) family
- Four feature engineering modes (composite, sma_only, variance_only, magnitude_only)

### Changed
- Migrated from hardcoded settings to centralized YAML configuration
- Replaced multiple entry points with single `run_production.py`
- Updated `preprocessing.create_variables()` to require `hour_bins` parameter from config
- Improved error handling and validation (fail-fast on missing config)

### Removed
- Hardcoded `NUM_HOUR_BINS` constant from `src/settings.py`
- Deprecated scripts: `run_ais_guarded.py`, `run_ais_scan.py`, `run_proposal_pipeline.py`, `run_smoke_corrected.py`
- Legacy compliance and process documentation files

### Fixed
- CTE hour_bins now correctly uses configured value instead of hardcoded 24
- Config validation enforces required fields before processing starts
- K-selection tracking now includes `k_original` and `capped` status

## [1.0.0] - 2025-01-26

### Initial Release
- Basic TE/CTE/STE/GC pipeline
- ExtraSensory dataset support
- JIDT integration for discrete TE computation
- Fixed k=4 configuration
- 24-hour CTE stratification
