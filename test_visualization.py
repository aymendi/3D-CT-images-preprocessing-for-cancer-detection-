#!/usr/bin/env python3
"""
Test script to demonstrate visualization capabilities
===================================================

This script creates a synthetic lung MRI volume and runs the preprocessing pipeline
to generate visualization examples.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import shutil

# Import pipeline functions
from pipeline import (
    load_nifti, save_nifti, resample_isotropic, clip_intensities,
    normalize_volume, segment_lungs_simple, create_qc_visualization,
    create_3d_visualization, run_pipeline
)


def create_synthetic_lung_mri():
    """Create a synthetic lung MRI volume for testing."""
    # Create a 3D volume with realistic dimensions
    volume = np.random.randn(128, 128, 128).astype(np.float32)
    
    # Add lung-like structure (darker regions)
    # Left lung
    volume[40:88, 40:88, 40:88] -= 3.0
    # Right lung  
    volume[40:88, 40:88, 88:128] -= 3.0
    
    # Add some anatomical structures
    # Heart (brighter region)
    volume[60:80, 60:80, 60:80] += 2.0
    
    # Add noise
    volume += np.random.randn(128, 128, 128) * 0.2
    
    # Add some artifacts
    volume[0:10, :, :] += 1.0  # Top slice artifact
    volume[:, 0:10, :] += 0.5  # Side artifact
    
    return volume


def main():
    """Run the visualization test."""
    print("Creating synthetic lung MRI volume...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir()
    
    # Create synthetic volume
    volume = create_synthetic_lung_mri()
    
    # Create affine matrix (1.5mm isotropic spacing)
    affine = np.eye(4)
    affine[0, 0] = 1.5  # x spacing
    affine[1, 1] = 1.5  # y spacing
    affine[2, 2] = 1.5  # z spacing
    
    # Save as NIfTI
    volume_xyz = np.transpose(volume, (2, 1, 0))  # Convert to (X,Y,Z) for nibabel
    img = nib.Nifti1Image(volume_xyz, affine)
    nifti_path = input_dir / "synthetic_lung.nii.gz"
    nib.save(img, nifti_path)
    
    print(f"Saved synthetic volume to: {nifti_path}")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Create config file
    config_content = """
target_spacing_mm: 2.0
clip_percentiles: [1.0, 99.0]
normalization: "zscore"
do_lung_crop: true
bbox_margin_mm: 5
random_seed: 42
generate_qc_images: true
"""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print("\nRunning preprocessing pipeline...")
    
    # Run the pipeline
    run_pipeline(str(input_dir), str(output_dir), str(config_path))
    
    # Check outputs
    print(f"\nOutput directory: {output_dir}")
    
    volumes_dir = output_dir / "volumes"
    masks_dir = output_dir / "masks"
    qc_dir = output_dir / "qc"
    
    print(f"Volumes: {list(volumes_dir.glob('*.nii*'))}")
    print(f"Masks: {list(masks_dir.glob('*.nii*'))}")
    print(f"QC images: {list(qc_dir.glob('*.png'))}")
    
    # Copy results to current directory for easy viewing
    results_dir = Path("visualization_results")
    results_dir.mkdir(exist_ok=True)
    
    print(f"\nCopying results to {results_dir}...")
    
    # Copy QC images
    for qc_file in qc_dir.glob("*.png"):
        shutil.copy2(qc_file, results_dir / qc_file.name)
        print(f"  Copied: {qc_file.name}")
    
    # Copy processed volume
    for vol_file in volumes_dir.glob("*.nii*"):
        shutil.copy2(vol_file, results_dir / vol_file.name)
        print(f"  Copied: {vol_file.name}")
    
    # Copy mask
    for mask_file in masks_dir.glob("*.nii*"):
        shutil.copy2(mask_file, results_dir / mask_file.name)
        print(f"  Copied: {mask_file.name}")
    
    print(f"\n✅ Visualization test completed!")
    print(f"📁 Results saved to: {results_dir.absolute()}")
    print(f"🖼️  Open the PNG files to see the before/after comparisons")
    
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()
