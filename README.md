# 3D Lung CT Preprocessing Pipeline for Cancer Segmentation

Pipeline de prétraitement pour l'amélioration de la qualité d'un jeu de données 3D de poumons pour la segmentation du cancer.

**Conforme au Cahier des Charges:**
- Analyse exploratoire du dataset
- Techniques de filtrage et amélioration de qualité (Gauss, médian, bilateral, TV, CLAHE, histogram stretching)
- Comparaison quantitative (PSNR, SSIM)
- Pipeline automatisé réutilisable

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline

```bash
# Basic usage
python pipeline.py --input data/Task06_Lung/imagesTr --output data/processed

# With custom config
python pipeline.py --input data/raw --output data/processed --config configs/pipeline.yaml
```

### 3. Check Output

```
data/processed/
├── volumes/          # Preprocessed volumes
│   ├── lung_001.nii.gz
│   └── lung_003.nii.gz
├── masks/            # Lung masks (if cropping enabled)
│   ├── lung_001_mask.nii.gz
│   └── lung_003_mask.nii.gz
├── qc/               # Quality control visualizations
│   ├── lung_001_comparison.png      # Before/after comparison
│   ├── lung_001_3d_original.png     # Original volume 3D view
│   ├── lung_001_3d_processed.png    # Processed volume 3D view
│   └── lung_003_comparison.png
└── metrics/          # Quality metrics (PSNR, SSIM)
    ├── lung_001_metrics.json
    └── lung_003_metrics.json
```

## Configuration

Edit `configs/pipeline.yaml` to customize:

```yaml
# Target spacing for resampling (mm)
target_spacing_mm: 2.0

# Intensity clipping percentiles [low, high]
clip_percentiles: [1.0, 99.0]

# Normalization method: "zscore" or "minmax"
normalization: "zscore"

# Lung cropping options
do_lung_crop: true
bbox_margin_mm: 10  # Margin around lung bounding box (mm)

# Reproducibility
random_seed: 42

# Visualization options
generate_qc_images: true  # Generate quality control visualizations
```

## Pipeline Steps

1. **Load**: Read NIfTI files with proper orientation handling
2. **Resample**: Resample to target spacing (default: 2.0mm isotropic)
3. **Clip**: Clip intensities using robust percentiles (1st-99th)
4. **Normalize**: Z-score or min-max normalization
5. **Crop**: Optional lung segmentation and bounding box cropping
6. **Save**: Save preprocessed volumes and masks
7. **Visualize**: Generate quality control images for before/after comparison

## Input/Output Layout

### Input
- Directory containing NIfTI files (`.nii` or `.nii.gz`)
- Example: `data/Task06_Lung/imagesTr/lung_001.nii.gz`

### Output
- `volumes/`: Preprocessed volumes with target spacing and normalized intensities
- `masks/`: Lung masks (if `do_lung_crop: true`)
- `qc/`: Quality control visualizations (if `generate_qc_images: true`)
  - `*_comparison.png`: Before/after comparison with multiple views
  - `*_3d_original.png`: 3D visualization of original volume
  - `*_3d_processed.png`: 3D visualization of processed volume with mask overlay

## CLI Options

```bash
python pipeline.py --help

Options:
  -i, --input DIR     Directory containing NIfTI files to process
  -o, --output DIR    Directory to save processed files
  -c, --config FILE   Path to YAML configuration file (default: configs/pipeline.yaml)
```

## Examples

```bash
# Process all NIfTI files in a directory
python pipeline.py --input data/Task06_Lung/imagesTr --output data/processed

# Use custom configuration
python pipeline.py --input data/raw --output data/processed --config my_config.yaml

# Run filter comparison (Cahier des Charges requirement)
python filter_comparison.py --input data/Task06_Lung/imagesTr/lung_001.nii.gz --output results/

# Test with synthetic data
python test_visualization.py
```

## 3D Tumor Segmentation

To evaluate the impact of the preprocessing pipeline, a 3D U-Net model was trained to segment lung tumors from the preprocessed CT scans.

### Model Architecture

The segmentation model is a 3D U-Net with the following features:
-   **Base Channels**: 32
-   **Depth**: 4
-   **Residual Connections**: Enabled
-   **Attention Mechanisms**: Enabled
-   **Deep Supervision**: Enabled
-   **Dropout**: 0.1

This architecture is defined in `segmentation/models/unet3d.py` and configured in `configs/seg3d.yaml`.

### Training

The model was trained on the `Task06_Lung` dataset with the following configuration:
-   **Loss Function**: A combination of Dice Loss (60%), Cross-Entropy (20%), and Focal Loss (20%).
-   **Optimizer**: Adam with a learning rate of `1e-3` and weight decay of `1e-5`.
-   **Learning Rate Schedule**: Cosine annealing.
-   **Epochs**: 200, with early stopping after 30 epochs of no improvement.
-   **Data Augmentation**: Random flips on all axes.
-   **Patches**: 50% of patches were centered on foreground (tumor) voxels.

### Evaluation

The model was evaluated on a validation set, achieving the following average metrics for the tumor class:
-   **Dice Score**: 0.04
-   **IoU (Jaccard)**: 0.02

**Note**: The low Dice score suggests that the model struggles with this dataset. This could be due to the small size of the tumors, the limited number of training examples, or the need for further hyperparameter tuning. The detailed per-case metrics can be found in `results/segmentations/metrics/val_metrics.csv`.

### How to Run Segmentation

#### 1. Train the Model

```bash
python segmentation/train.py --config configs/seg3d.yaml
```

#### 2. Run Inference

To generate segmentations on new images, use the `infer.py` script with a trained model checkpoint:

```bash
python segmentation/infer.py --config configs/seg3d.yaml --checkpoint checkpoints/final_model.pth --input data/Task06_Lung/imagesTs --output results/segmentations/infer
```

#### 3. Evaluate Metrics

To evaluate the performance of a trained model on a labeled dataset:

```bash
python segmentation/evaluate.py --config configs/seg3d.yaml --checkpoint checkpoints/final_model.pth --input data/Task06_Lung/imagesTs --output results/segmentations/metrics
```

## Requirements

- Python 3.8+
- nibabel (NIfTI I/O)
- SimpleITK (resampling)
- numpy, scipy (numerical operations)
- scikit-image (image processing)
- matplotlib (visualization)
- PyYAML (configuration)

## Visualization Features

The pipeline automatically generates quality control visualizations to help you compare before and after processing:

### Comparison Images (`*_comparison.png`)
- **Side-by-side views**: Original vs processed slices at multiple Z positions
- **Mask overlay**: Shows lung segmentation results in red
- **Difference maps**: Highlights changes between original and processed
- **Intensity histograms**: Compare intensity distributions
- **Summary statistics**: Volume dimensions, spacing, and intensity ranges

### 3D Visualizations
- **Original volume** (`*_3d_original.png`): Axial, coronal, and sagittal views
- **Processed volume** (`*_3d_processed.png`): Same views with lung mask overlay
- **Volume statistics**: Shape, intensity ranges, and mask coverage

### Example Visualization Layout
```
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   Original      │   Processed     │  Mask Overlay   │   Histograms    │
│   (Z=25)        │   (Z=25)        │   (Z=25)        │   (Z=25)        │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│   Original      │   Processed     │  Difference     │   Histograms    │
│   (Z=50)        │   (Z=50)        │   (Z=50)        │   (Z=50)        │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│   Original      │   Processed     │  Difference     │   Histograms    │
│   (Z=75)        │   (Z=75)        │   (Z=75)        │   (Z=75)        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## Notes

- Pipeline preserves original affine matrices for proper spatial alignment
- All operations are deterministic (set `random_seed` for reproducibility)
- Robust error handling with fallbacks for failed operations
- Memory-efficient processing of large 3D volumes
- High-quality visualizations (150 DPI) for detailed inspection

## Project Structure

The project has been cleaned to contain only essential files:

```
.
├─ pipeline.py              # Main preprocessing pipeline
├─ configs/
│  └─ pipeline.yaml         # Configuration file
├─ data/
│  └─ Task06_Lung/          # Your lung dataset
│      ├─ imagesTr/         # Training images
│      ├─ imagesTs/         # Test images
│      └─ labelsTr/         # Training labels
├─ test_visualization.py    # Test script with synthetic data
├─ requirements.txt         # Dependencies
└─ README.md               # This file
```

**Files included:**
- `pipeline.py` - Pipeline principal avec filtrage et métriques
- `filter_comparison.py` - Script de comparaison des filtres (Cahier des Charges)
- `lung_preprocessing_analysis.ipynb` - Notebook d'analyse complet
- `test_visualization.py` - Script de test avec données synthétiques
- `configs/pipeline.yaml` - Configuration avec options de filtrage

**Removed unnecessary files:**
- Complex source code modules (consolidated into single pipeline.py)
- Old results and derivatives
- Complex test suites
- Build configuration files