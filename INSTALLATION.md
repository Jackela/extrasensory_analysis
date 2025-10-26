# Installation Guide

## Prerequisites

- **Python**: 3.12 or higher
- **Java**: 8 or higher (for JIDT)
- **RAM**: 12GB minimum (48GB recommended for parallel execution)
- **Storage**: ~5GB for dataset + outputs

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/extrasensory_analysis.git
cd extrasensory_analysis
```

## Step 2: Python Environment Setup

### Option A: Using pip (Recommended)

```bash
pip install -r requirements.txt
```

### Option B: Using conda

```bash
conda create -n extrasensory python=3.12
conda activate extrasensory
pip install -r requirements.txt
```

## Step 3: Download JIDT

**JIDT (Java Information Dynamics Toolkit)** is required for transfer entropy computation.

1. **Download JIDT v1.5+**:
   - Visit: https://github.com/jlizier/jidt/releases
   - Download the latest release (e.g., `infodynamics-dist-1.6.1.zip`)

2. **Extract and setup**:
   ```bash
   # Create jidt directory
   mkdir -p jidt
   
   # Extract the downloaded file
   unzip infodynamics-dist-1.6.1.zip -d jidt/
   
   # Copy the jar file to project root
   cp jidt/infodynamics-dist-1.6.1/infodynamics.jar jidt/
   ```

3. **Verify installation**:
   ```bash
   java -jar jidt/infodynamics.jar
   # Should show JIDT information
   ```

**Expected structure**:
```
extrasensory_analysis/
├── jidt/
│   └── infodynamics.jar    # Required
```

## Step 4: Download ExtraSensory Dataset

**ExtraSensory Dataset** provides labeled smartphone sensor data.

1. **Download**:
   - Visit: http://extrasensory.ucsd.edu/
   - Download `ExtraSensory.per_uuid_features_labels.zip` (~2GB)

2. **Extract**:
   ```bash
   # Extract to data directory
   mkdir -p data
   unzip ExtraSensory.per_uuid_features_labels.zip -d data/
   ```

3. **Verify structure**:
   ```bash
   ls data/ExtraSensory.per_uuid_features_labels/*.gz | wc -l
   # Should show 60 files (one per user UUID)
   ```

**Expected structure**:
```
extrasensory_analysis/
├── data/
│   └── ExtraSensory.per_uuid_features_labels/
│       ├── 00EABED2-271D-49D8-B599-1D4A09240601.features_labels.csv.gz
│       ├── 0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv.gz
│       └── ... (60 users total)
```

## Step 5: Verify Installation

Run smoke test to verify setup:

```bash
python run_production.py smoke
```

**Expected output**:
```
PRESET: smoke → config/presets/smoke.yaml
Loading 2 users...
Running k-selection (fixed k=4)...
Computing TE...
✓ Smoke test completed successfully
```

## Troubleshooting

### Java not found
```bash
# Install Java 8+
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Windows
# Download from https://adoptium.net/
```

### JIDT jar not found
```bash
# Verify jar exists
ls -lh jidt/infodynamics.jar

# If missing, re-download JIDT (see Step 3)
```

### Dataset not found
```bash
# Verify dataset structure
ls data/ExtraSensory.per_uuid_features_labels/ | head -5

# If missing, re-download dataset (see Step 4)
```

### Out of memory
```bash
# Check available RAM
free -h  # Linux
vm_stat  # macOS

# Reduce user count for testing
python run_production.py smoke --n-users 1
```

## Next Steps

See [QUICK_START.md](QUICK_START.md) for usage examples and preset configurations.
