# JIDT Directory

This directory is for the **Java Information Dynamics Toolkit (JIDT)**.

## Download Instructions

1. **Visit**: https://github.com/jlizier/jidt/releases

2. **Download**: Latest release (e.g., `infodynamics-dist-1.6.1.zip`)

3. **Extract and setup**:
   ```bash
   # Extract downloaded file
   unzip infodynamics-dist-1.6.1.zip -d jidt/
   
   # Copy jar to project root
   cp jidt/infodynamics-dist-1.6.1/infodynamics.jar jidt/
   ```

4. **Expected structure**:
   ```
   jidt/
   └── infodynamics.jar    # Required for analysis
   ```

## JIDT Information

- **Source**: Java Information Dynamics Toolkit
- **Author**: Joseph T. Lizier (University of Sydney)
- **Version**: 1.5+ (1.6.1 recommended)
- **License**: GPL-3.0
- **Purpose**: Information-theoretic measures (TE, MI, AIS, etc.)

## Usage in This Project

JIDT is used for:
- **Transfer Entropy (TE)**: Directed information flow computation
- **Active Information Storage (AIS)**: Optimal k-selection
- **Conditional Transfer Entropy (CTE)**: Hour-of-day stratified analysis

## Citation

```bibtex
@article{lizier2014jidt,
  title={JIDT: An information-theoretic toolkit for studying the dynamics of complex systems},
  author={Lizier, Joseph T},
  journal={Frontiers in Robotics and AI},
  volume={1},
  pages={11},
  year={2014}
}
```

## Verification

Test JIDT installation:

```bash
java -jar jidt/infodynamics.jar
# Should display JIDT information and version
```

## Note

**This directory is excluded from git** (see `.gitignore`). Users must download JIDT themselves to comply with GPL-3.0 license.
