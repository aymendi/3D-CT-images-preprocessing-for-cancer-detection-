#!/usr/bin/env python3
"""
Filter Comparison Script - Cahier des Charges Requirements
=========================================================

Script pour comparer différents filtres et techniques d'amélioration
selon les spécifications du Cahier des Charges.

Ce script:
1. Applique différents filtres de débruitage
2. Applique différentes techniques d'amélioration de contraste
3. Calcule les métriques de qualité (PSNR, SSIM)
4. Génère des visualisations comparatives
5. Sauvegarde les résultats pour analyse
"""

import numpy as np
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline import (
    load_nifti, save_nifti, resample_isotropic, clip_intensities,
    normalize_volume, apply_denoising_filters, apply_contrast_enhancement,
    compare_filter_effects, create_qc_visualization
)


def run_filter_comparison(input_file: str, output_dir: str, config_path: str = "configs/pipeline.yaml"):
    """
    Run comprehensive filter comparison as specified in Cahier des Charges.
    
    Parameters
    ----------
    input_file : str
        Path to input NIfTI file
    output_dir : str
        Output directory for results
    config_path : str
        Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Create output directories
    output_path = Path(output_dir)
    comparison_dir = output_path / "filter_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original volume
    print(f"Loading volume: {input_file}")
    original_volume, affine, spacing = load_nifti(input_file)
    original_volume = np.nan_to_num(original_volume, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Basic preprocessing (resample, clip, normalize)
    target_spacing = config.get('target_spacing_mm', 2.0)
    if isinstance(target_spacing, (int, float)):
        target_spacing = (target_spacing, target_spacing, target_spacing)
    
    volume, spacing = resample_isotropic(original_volume, spacing, target_spacing)
    volume = clip_intensities(volume, tuple(config.get('clip_percentiles', [1.0, 99.0])))
    volume = normalize_volume(volume, config.get('normalization', 'zscore'))
    
    print(f"Preprocessed volume shape: {volume.shape}")
    
    # Define filter combinations to test
    denoising_methods = ['gaussian', 'median', 'bilateral', 'tv']
    contrast_methods = ['clahe', 'histogram_stretch', 'gamma']
    
    # Store results
    results = []
    
    # Test each combination
    for denoise_method in denoising_methods:
        for contrast_method in contrast_methods:
            print(f"\nTesting combination: {denoise_method} + {contrast_method}")
            
            # Apply denoising
            denoised = apply_denoising_filters(volume, denoise_method)
            
            # Apply contrast enhancement
            enhanced = apply_contrast_enhancement(denoised, contrast_method)
            
            # Calculate metrics
            metrics = compare_filter_effects(volume, enhanced, f"{denoise_method}_{contrast_method}")
            results.append(metrics)
            
            # Save processed volume
            output_file = comparison_dir / f"processed_{denoise_method}_{contrast_method}.nii.gz"
            save_nifti(str(output_file), enhanced, affine)
            
            # Create comparison visualization
            viz_file = comparison_dir / f"comparison_{denoise_method}_{contrast_method}.png"
            create_qc_visualization(
                original_volume=volume,
                processed_volume=enhanced,
                mask=None,
                original_spacing=spacing,
                processed_spacing=spacing,
                output_path=str(viz_file)
            )
            
            print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.3f}")
    
    # Save comprehensive results
    results_file = comparison_dir / "filter_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary visualization
    create_summary_plot(results, str(comparison_dir / "filter_comparison_summary.png"))
    
    # Find best combinations
    best_psnr = max(results, key=lambda x: x['psnr'])
    best_ssim = max(results, key=lambda x: x['ssim'])
    
    print(f"\n=== FILTER COMPARISON RESULTS ===")
    print(f"Best PSNR: {best_psnr['filter_name']} - {best_psnr['psnr']:.2f} dB")
    print(f"Best SSIM: {best_ssim['filter_name']} - {best_ssim['ssim']:.3f}")
    print(f"\nResults saved to: {comparison_dir}")
    
    return results


def create_summary_plot(results: list, output_path: str):
    """Create a summary plot of all filter combinations."""
    # Extract data
    filter_names = [r['filter_name'] for r in results]
    psnr_values = [r['psnr'] for r in results]
    ssim_values = [r['ssim'] for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # PSNR plot
    bars1 = ax1.bar(range(len(filter_names)), psnr_values, color='skyblue', alpha=0.7)
    ax1.set_title('PSNR Comparison - Filter Combinations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_xticks(range(len(filter_names)))
    ax1.set_xticklabels(filter_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, psnr_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # SSIM plot
    bars2 = ax2.bar(range(len(filter_names)), ssim_values, color='lightcoral', alpha=0.7)
    ax2.set_title('SSIM Comparison - Filter Combinations', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SSIM')
    ax2.set_xlabel('Filter Combinations')
    ax2.set_xticks(range(len(filter_names)))
    ax2.set_xticklabels(filter_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, ssim_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {output_path}")


def main():
    """Main function to run filter comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter Comparison for Lung CT Preprocessing")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input NIfTI file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--config", "-c", type=str, default="configs/pipeline.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_filter_comparison(args.input, args.output, args.config)
    
    print("\n✅ Filter comparison completed!")
    print("📊 Check the output directory for detailed results and visualizations")


if __name__ == "__main__":
    main()
