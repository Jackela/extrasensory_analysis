# Data Directory

This directory is for the **ExtraSensory Dataset**.

## Download Instructions

1. **Visit**: http://extrasensory.ucsd.edu/

2. **Download**: `ExtraSensory.per_uuid_features_labels.zip` (~2GB)

3. **Extract here**:
   ```bash
   unzip ExtraSensory.per_uuid_features_labels.zip -d data/
   ```

4. **Expected structure**:
   ```
   data/
   └── ExtraSensory.per_uuid_features_labels/
       ├── 00EABED2-271D-49D8-B599-1D4A09240601.features_labels.csv.gz
       ├── 0A986513-7828-4D53-AA1F-E02D6DF9561B.features_labels.csv.gz
       └── ... (60 users total)
   ```

## Dataset Information

- **Source**: ExtraSensory - Human Activity and Context Recognition Dataset
- **Paper**: Vaizman et al. (2017), IEEE Pervasive Computing
- **Size**: ~2GB compressed
- **Users**: 60 participants
- **Duration**: Days to weeks per user
- **Sensors**: Accelerometer, gyroscope, location, audio, etc.
- **Labels**: 51 activity and context labels

## Citation

```bibtex
@article{vaizman2017recognizing,
  title={Recognizing Detailed Human Context in the Wild from Smartphones and Smartwatches},
  author={Vaizman, Yonatan and Ellis, Katherine and Lanckriet, Gert},
  journal={IEEE Pervasive Computing},
  volume={16},
  number={4},
  pages={62--74},
  year={2017}
}
```

## Note

**This directory is excluded from git** (see `.gitignore`). Users must download the dataset themselves.
