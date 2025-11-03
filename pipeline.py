#!/usr/bin/env python3
"""
3D Lung MRI Preprocessing Pipeline for Cancer Segmentation
========================================================

Pipeline de prétraitement pour l'amélioration de la qualité d'un jeu de données 3D 
de poumons pour la segmentation du cancer.

Conforme au Cahier des Charges:
- Analyse exploratoire du dataset
- Techniques de filtrage et amélioration de qualité
- Comparaison quantitative (PSNR, SSIM)
- Pipeline automatisé réutilisable

Usage:
    python pipeline.py --input data/Task06_Lung/imagesTr --output data/processed
    python pipeline.py --input data/raw --output data/processed --config configs/pipeline.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball, disk, remove_small_holes, remove_small_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.filters import gaussian, median


# =============================================================================
# Core Functions (adapted from existing codebase)
# =============================================================================

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI file and return (volume_zyx, affine, spacing_zyx)."""
    img = nib.load(path)
    affine = img.affine
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("NIfTI volume must be 3D (or 4D with first volume taken)")
    vol_zyx = np.transpose(data, (2, 1, 0)).astype(np.float32, copy=False)
    spacing_zyx = _affine_spacing_zyx(img.affine)
    return vol_zyx, affine, spacing_zyx


def save_nifti(path: str, vol_zyx: np.ndarray, affine: np.ndarray) -> None:
    """Save a (Z,Y,X) volume as NIfTI with a given affine."""
    if vol_zyx.ndim != 3:
        raise ValueError("Volume must be 3D (Z,Y,X)")
    data_xyz = np.transpose(vol_zyx, (2, 1, 0))
    img = nib.Nifti1Image(data_xyz, affine)
    nib.save(img, path)


def _affine_spacing_zyx(affine: np.ndarray) -> Tuple[float, float, float]:
    """Extract voxel spacing from affine matrix as (z,y,x)."""
    spacing_xyz = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])


def resample_isotropic(
    volume: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    target_zyx: Tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Resample a 3D volume to target spacing using linear interpolation."""
    if target_zyx is None or any(t <= 0 for t in target_zyx):
        return volume.astype(np.float32, copy=False), spacing_zyx
    
    try:
        img = sitk.GetImageFromArray(volume.astype(np.float32))
        img.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
        orig_size = np.array(list(reversed(volume.shape)), dtype=float)
        orig_spacing = np.array([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]], dtype=float)
        target_spacing = np.array([target_zyx[2], target_zyx[1], target_zyx[0]], dtype=float)
        new_size_xyz = np.rint(orig_size * (orig_spacing / target_spacing)).astype(int)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(tuple(target_spacing.tolist()))
        resampler.SetSize(tuple(int(x) for x in new_size_xyz.tolist()))
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampled = resampler.Execute(img)
        out = sitk.GetArrayFromImage(resampled).astype(np.float32)
        return out, (float(target_zyx[0]), float(target_zyx[1]), float(target_zyx[2]))
    except Exception:
        # SciPy fallback
        sz, sy, sx = spacing_zyx
        zoom_factors = (
            float(sz) / float(target_zyx[0]),
            float(sy) / float(target_zyx[1]),
            float(sx) / float(target_zyx[2]),
        )
        out = ndi.zoom(volume.astype(np.float32), zoom=zoom_factors, order=1)
        return out.astype(np.float32, copy=False), (
            float(target_zyx[0]),
            float(target_zyx[1]),
            float(target_zyx[2]),
        )


def clip_intensities(volume: np.ndarray, percentiles: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Clip intensities using robust percentiles."""
    vol = volume.astype(np.float32, copy=False)
    finite = np.isfinite(vol)
    if not finite.any():
        return np.zeros_like(vol)
    
    lo, hi = np.percentile(vol[finite], percentiles)
    if hi <= lo:
        return np.clip((vol - lo), 0, None)
    
    clipped = np.clip(vol, lo, hi)
    return clipped.astype(np.float32, copy=False)


def normalize_volume(volume: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize volume using z-score or min-max scaling."""
    vol = volume.astype(np.float32, copy=False)
    finite = np.isfinite(vol)
    if not finite.any():
        return np.zeros_like(vol)
    
    if method == "zscore":
        mean_val = vol[finite].mean()
        std_val = vol[finite].std()
        if std_val < 1e-6:
            std_val = 1.0
        normalized = (vol - mean_val) / std_val
    elif method == "minmax":
        min_val = vol[finite].min()
        max_val = vol[finite].max()
        if max_val <= min_val:
            return np.zeros_like(vol)
        normalized = (vol - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.float32, copy=False)



def rescale_intensity_percentile(vol: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Rescale intensities of a volume based on lower and upper percentiles.
    Output is scaled to [0, 1] after clipping.
    """
    v_min = np.percentile(vol, p_low)
    v_max = np.percentile(vol, p_high)
    vol_clipped = np.clip(vol, v_min, v_max)
    return (vol_clipped - v_min) / (v_max - v_min + 1e-8)




def segment_lungs_simple(volume: np.ndarray) -> np.ndarray:
    """Simple lung segmentation using Otsu thresholding and morphology."""
    vol = volume.astype(np.float32, copy=False)
    Z, Y, X = vol.shape
    mask = np.zeros_like(vol, dtype=bool)
    
    # Per-slice Otsu thresholding
    for z in range(Z):
        slc = vol[z]
        finite = np.isfinite(slc)
        if not finite.any():
            continue
            
        # Normalize slice to [0,1]
        slc_norm = (slc - slc[finite].min()) / (slc[finite].max() - slc[finite].min() + 1e-6)
        
        try:
            thr = threshold_otsu(slc_norm)
            lung2d = slc_norm < thr  # darker class is lung
        except Exception:
            lung2d = slc_norm < 0.5
        
        # Basic morphology
        lung2d = ndi.binary_opening(lung2d, structure=disk(2))
        lung2d = remove_small_objects(lung2d, min_size=128)
        lung2d = remove_small_holes(lung2d, area_threshold=128)
        mask[z] = lung2d
    
    # 3D cleanup
    mask = ndi.binary_closing(mask, structure=ball(1))
    
    # Keep largest 1-2 components
    labeled, num = ndi.label(mask)
    if num <= 2:
        final = mask
    else:
        sizes = ndi.sum(np.ones_like(mask, dtype=np.int32), labeled, index=list(range(1, num + 1)))
        idx_sorted = np.argsort(sizes)[::-1][:2]
        keep = np.isin(labeled, (idx_sorted + 1))
        final = keep
    
    final = remove_small_objects(final, min_size=5000)
    return final.astype(np.uint8)


def compute_bbox_from_mask(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    """Compute tight bounding box slices for a binary mask in (Z,Y,X)."""
    assert mask.ndim == 3, "Mask must be 3D (Z,Y,X)"
    z_any = mask.any(axis=(1, 2))
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))
    
    if not (z_any.any() and y_any.any() and x_any.any()):
        z, y, x = mask.shape
        return slice(0, z), slice(0, y), slice(0, x)
    
    zz = np.where(z_any)[0]
    yy = np.where(y_any)[0]
    xx = np.where(x_any)[0]
    return slice(zz[0], zz[-1] + 1), slice(yy[0], yy[-1] + 1), slice(xx[0], xx[-1] + 1)


def expand_bbox_with_margin(
    bbox: Tuple[slice, slice, slice],
    margin_vox_zyx: Tuple[int, int, int],
    shape_zyx: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """Expand bbox slices by integer voxel margins per axis with clamping."""
    (sz, sy, sx) = bbox
    mz, my, mx = margin_vox_zyx
    Z, Y, X = shape_zyx
    
    start_z = max(0, (sz.start or 0) - mz)
    start_y = max(0, (sy.start or 0) - my)
    start_x = max(0, (sx.start or 0) - mx)
    stop_z = min(Z, (sz.stop or Z) + mz)
    stop_y = min(Y, (sy.stop or Y) + my)
    stop_x = min(X, (sx.stop or X) + mx)
    
    return slice(start_z, stop_z), slice(start_y, stop_y), slice(start_x, stop_x)


def mm_to_vox(margin_mm_zyx: Tuple[float, float, float], spacing_zyx: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert margins in millimeters (z,y,x) to integer voxels."""
    margin_mm = np.asarray(margin_mm_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    voxels = np.rint(margin_mm / np.maximum(spacing, 1e-6)).astype(int)
    return int(voxels[0]), int(voxels[1]), int(voxels[2])


def update_affine_for_bbox(affine: np.ndarray, bbox: Tuple[slice, slice, slice]) -> np.ndarray:
    """Update the NIfTI affine matrix for a cropped bounding box."""
    if affine is None or affine.shape != (4, 4):
        return affine
    
    new_affine = affine.copy()
    z0, y0, x0 = bbox[0].start or 0, bbox[1].start or 0, bbox[2].start or 0
    
    # Shift the origin by the voxel offset in world coordinates
    offset_vox_h = np.array([x0, y0, z0, 1.0], dtype=np.float64)
    new_affine[:3, 3] = (affine @ offset_vox_h)[:3]
    return new_affine


# =============================================================================
# Quality Metrics Functions (Cahier des Charges Requirements)
# =============================================================================

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between original and processed images."""
    try:
        # Ensure both arrays have the same shape
        if original.shape != processed.shape:
            # Resize to match if needed
            from scipy.ndimage import zoom
            zoom_factors = tuple(processed.shape[i] / original.shape[i] for i in range(len(original.shape)))
            original = zoom(original, zoom_factors, order=1)
        
        # Normalize to [0, 1] range for PSNR calculation
        orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-6)
        proc_norm = (processed - processed.min()) / (processed.max() - processed.min() + 1e-6)
        
        psnr = peak_signal_noise_ratio(orig_norm, proc_norm, data_range=1.0)
        return float(psnr)
    except Exception as e:
        print(f"Warning: PSNR calculation failed: {e}")
        return 0.0


def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) between original and processed images."""
    try:
        # Ensure both arrays have the same shape
        if original.shape != processed.shape:
            # Resize to match if needed
            from scipy.ndimage import zoom
            zoom_factors = tuple(processed.shape[i] / original.shape[i] for i in range(len(original.shape)))
            original = zoom(original, zoom_factors, order=1)
        
        # Normalize to [0, 1] range for SSIM calculation
        orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-6)
        proc_norm = (processed - processed.min()) / (processed.max() - processed.min() + 1e-6)
        
        # Calculate SSIM for 3D volume (slice by slice and average)
        ssim_values = []
        for z in range(min(orig_norm.shape[0], proc_norm.shape[0])):
            ssim_val = structural_similarity(
                orig_norm[z], proc_norm[z], 
                data_range=1.0, 
                channel_axis=None
            )
            ssim_values.append(ssim_val)
        
        return float(np.mean(ssim_values))
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        return 0.0


def apply_denoising_filters(volume: np.ndarray, method: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Apply different denoising filters as specified in Cahier des Charges.
    
    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume
    method : str
        Denoising method: 'gaussian', 'median', 'bilateral', 'tv'
    **kwargs
        Additional parameters for the filter
        
    Returns
    -------
    np.ndarray
        Denoised volume
    """
    vol = volume.astype(np.float32, copy=False)
    
    if method == "gaussian":
        sigma = kwargs.get('sigma', 1.0)
        return gaussian(vol, sigma=sigma, preserve_range=True)
    
    elif method == "median":
        size = kwargs.get('size', 3)
        result = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            result[z] = median(vol[z], footprint=disk(size))
        return result
    
    elif method == "bilateral":
        sigma_color = kwargs.get('sigma_color', 0.05)
        sigma_spatial = kwargs.get('sigma_spatial', 2.0)
        result = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            result[z] = denoise_bilateral(
                vol[z], 
                sigma_color=sigma_color, 
                sigma_spatial=sigma_spatial,
                channel_axis=None
            )
        return result
    
    elif method == "tv":
        weight = kwargs.get('weight', 0.05)
        result = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            result[z] = denoise_tv_chambolle(vol[z], weight=weight, channel_axis=None)
        return result
    
    else:
        print(f"Warning: Unknown denoising method '{method}', returning original volume")
        return vol


def apply_contrast_enhancement(volume: np.ndarray, method: str = "clahe", **kwargs) -> np.ndarray:
    """
    Apply contrast enhancement techniques as specified in Cahier des Charges.
    
    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume
    method : str
        Enhancement method: 'clahe', 'histogram_stretch', 'gamma'
    **kwargs
        Additional parameters for the enhancement
        
    Returns
    -------
    np.ndarray
        Enhanced volume
    """
    vol = volume.astype(np.float32, copy=False)
    
    if method == "clahe":
        clip_limit = kwargs.get('clip_limit', 0.01)
        nbins = kwargs.get('nbins', 256)
        result = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            # Normalize slice to [0, 1]
            slice_norm = (vol[z] - vol[z].min()) / (vol[z].max() - vol[z].min() + 1e-6)
            # Apply CLAHE
            from skimage import exposure
            enhanced = exposure.equalize_adapthist(slice_norm, clip_limit=clip_limit, nbins=nbins)
            # Scale back to original range
            result[z] = enhanced * (vol[z].max() - vol[z].min()) + vol[z].min()
        return result
    
    elif method == "histogram_stretch":
        p_low = kwargs.get('p_low', 1.0)
        p_high = kwargs.get('p_high', 99.0)
        return rescale_intensity_percentile(vol, p_low, p_high)
    
    elif method == "gamma":
        gamma = kwargs.get('gamma', 1.0)
        # Normalize to [0, 1]
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        # Apply gamma correction
        vol_gamma = np.power(vol_norm, gamma)
        # Scale back to original range
        return vol_gamma * (vol.max() - vol.min()) + vol.min()
    
    else:
        print(f"Warning: Unknown enhancement method '{method}', returning original volume")
        return vol


def compare_filter_effects(original: np.ndarray, processed: np.ndarray, filter_name: str) -> dict:
    """
    Compare the effects of different filters using quantitative metrics.
    
    Parameters
    ----------
    original : np.ndarray
        Original volume
    processed : np.ndarray
        Processed volume
    filter_name : str
        Name of the applied filter
        
    Returns
    -------
    dict
        Dictionary containing quality metrics
    """
    metrics = {
        'filter_name': filter_name,
        'psnr': calculate_psnr(original, processed),
        'ssim': calculate_ssim(original, processed),
        'original_shape': original.shape,
        'processed_shape': processed.shape,
        'original_range': [float(original.min()), float(original.max())],
        'processed_range': [float(processed.min()), float(processed.max())],
        'original_mean': float(original.mean()),
        'processed_mean': float(processed.mean()),
        'original_std': float(original.std()),
        'processed_std': float(processed.std())
    }
    
    return metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def create_qc_visualization(
    original_volume: np.ndarray,
    processed_volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    original_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    processed_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: str = "qc_comparison.png"
) -> None:
    """
    Create a quality control visualization showing before/after comparison.
    
    Parameters
    ----------
    original_volume : np.ndarray
        Original volume (Z,Y,X)
    processed_volume : np.ndarray
        Processed volume (Z,Y,X)
    mask : np.ndarray, optional
        Lung mask for overlay
    original_spacing : tuple
        Original volume spacing
    processed_spacing : tuple
        Processed volume spacing
    output_path : str
        Path to save the visualization
    """
    # Select representative slices (middle, quarter, three-quarter)
    z_orig = original_volume.shape[0]
    z_proc = processed_volume.shape[0]
    
    orig_slices = [z_orig // 4, z_orig // 2, 3 * z_orig // 4]
    proc_slices = [z_proc // 4, z_proc // 2, 3 * z_proc // 4]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Add title
    fig.suptitle('3D Lung MRI Preprocessing - Before vs After Comparison', fontsize=16, fontweight='bold')
    
    for i, (orig_idx, proc_idx) in enumerate(zip(orig_slices, proc_slices)):
        # Original slice
        ax_orig = fig.add_subplot(gs[i, 0])
        orig_slice = original_volume[orig_idx]
        im_orig = ax_orig.imshow(orig_slice, cmap='gray', aspect='equal')
        ax_orig.set_title(f'Original (Z={orig_idx})\nSpacing: {original_spacing[0]:.1f}mm', fontsize=10)
        ax_orig.axis('off')
        plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, pad=0.04)
        
        # Processed slice
        ax_proc = fig.add_subplot(gs[i, 1])
        proc_slice = processed_volume[proc_idx]
        im_proc = ax_proc.imshow(proc_slice, cmap='gray', aspect='equal')
        ax_proc.set_title(f'Processed (Z={proc_idx})\nSpacing: {processed_spacing[0]:.1f}mm', fontsize=10)
        ax_proc.axis('off')
        plt.colorbar(im_proc, ax=ax_proc, fraction=0.046, pad=0.04)
        
        # Overlay with mask if available
        if mask is not None and mask.shape[0] > proc_idx:
            ax_overlay = fig.add_subplot(gs[i, 2])
            ax_overlay.imshow(proc_slice, cmap='gray', aspect='equal')
            mask_slice = mask[proc_idx]
            # Create colored overlay for mask
            mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax_overlay.imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='equal')
            ax_overlay.set_title(f'Lung Mask Overlay\n(Z={proc_idx})', fontsize=10)
            ax_overlay.axis('off')
        else:
            # Show difference if no mask
            ax_diff = fig.add_subplot(gs[i, 2])
            # Resize original slice to match processed if needed
            if orig_slice.shape != proc_slice.shape:
                from scipy.ndimage import zoom
                zoom_factors = (proc_slice.shape[0] / orig_slice.shape[0], 
                              proc_slice.shape[1] / orig_slice.shape[1])
                orig_slice_resized = zoom(orig_slice, zoom_factors, order=1)
            else:
                orig_slice_resized = orig_slice
            
            # Normalize both slices for comparison
            orig_norm = (orig_slice_resized - orig_slice_resized.min()) / (orig_slice_resized.max() - orig_slice_resized.min() + 1e-6)
            proc_norm = (proc_slice - proc_slice.min()) / (proc_slice.max() - proc_slice.min() + 1e-6)
            diff = proc_norm - orig_norm
            
            im_diff = ax_diff.imshow(diff, cmap='RdBu_r', aspect='equal', vmin=-0.5, vmax=0.5)
            ax_diff.set_title(f'Difference (Processed - Original)\n(Z={proc_idx})', fontsize=10)
            ax_diff.axis('off')
            plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        
        # Histogram comparison
        ax_hist = fig.add_subplot(gs[i, 3])
        ax_hist.hist(orig_slice.flatten(), bins=50, alpha=0.7, label='Original', color='blue', density=True)
        ax_hist.hist(proc_slice.flatten(), bins=50, alpha=0.7, label='Processed', color='red', density=True)
        ax_hist.set_title(f'Intensity Distribution\n(Z={proc_idx})', fontsize=10)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    
    # Add summary statistics
    fig.text(0.02, 0.02, 
             f'Summary:\n'
             f'Original shape: {original_volume.shape}\n'
             f'Processed shape: {processed_volume.shape}\n'
             f'Original spacing: {original_spacing}\n'
             f'Processed spacing: {processed_spacing}\n'
             f'Original range: [{original_volume.min():.2f}, {original_volume.max():.2f}]\n'
             f'Processed range: [{processed_volume.min():.2f}, {processed_volume.max():.2f}]',
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Quality control visualization saved to: {output_path}")


def create_3d_visualization(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    output_path: str = "3d_visualization.png",
    title: str = "3D Volume Visualization"
) -> None:
    """
    Create a 3D visualization showing orthogonal views of the volume.
    
    Parameters
    ----------
    volume : np.ndarray
        Volume to visualize (Z,Y,X)
    mask : np.ndarray, optional
        Mask for overlay
    output_path : str
        Path to save the visualization
    title : str
        Title for the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Get middle slices
    z_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    x_mid = volume.shape[2] // 2
    
    # Axial view (Z slice)
    axes[0, 0].imshow(volume[z_mid], cmap='gray', aspect='equal')
    if mask is not None:
        mask_overlay = np.ma.masked_where(mask[z_mid] == 0, mask[z_mid])
        axes[0, 0].imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='equal')
    axes[0, 0].set_title(f'Axial View (Z={z_mid})')
    axes[0, 0].axis('off')
    
    # Coronal view (Y slice)
    axes[0, 1].imshow(volume[:, y_mid, :], cmap='gray', aspect='equal')
    if mask is not None:
        mask_overlay = np.ma.masked_where(mask[:, y_mid, :] == 0, mask[:, y_mid, :])
        axes[0, 1].imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='equal')
    axes[0, 1].set_title(f'Coronal View (Y={y_mid})')
    axes[0, 1].axis('off')
    
    # Sagittal view (X slice)
    axes[1, 0].imshow(volume[:, :, x_mid], cmap='gray', aspect='equal')
    if mask is not None:
        mask_overlay = np.ma.masked_where(mask[:, :, x_mid] == 0, mask[:, :, x_mid])
        axes[1, 0].imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='equal')
    axes[1, 0].set_title(f'Sagittal View (X={x_mid})')
    axes[1, 0].axis('off')
    
    # Volume statistics
    axes[1, 1].text(0.1, 0.9, f'Volume Statistics:', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f'Shape: {volume.shape}', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Min: {volume.min():.2f}', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Max: {volume.max():.2f}', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Mean: {volume.mean():.2f}', fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Std: {volume.std():.2f}', fontsize=10, transform=axes[1, 1].transAxes)
    
    if mask is not None:
        mask_volume = np.sum(mask > 0)
        total_volume = mask.size
        axes[1, 1].text(0.1, 0.3, f'Mask volume: {mask_volume:,} voxels', fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f'Mask coverage: {100*mask_volume/total_volume:.1f}%', fontsize=10, transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D visualization saved to: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(input_dir: str, output_dir: str, config_path: str = "configs/pipeline.yaml") -> None:
    """
    Run the complete preprocessing pipeline on all NIfTI files in input_dir.
    
    Parameters
    ----------
    input_dir : str
        Directory containing NIfTI files to process
    output_dir : str
        Directory to save processed files
    config_path : str
        Path to YAML configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    if config.get('random_seed') is not None:
        np.random.seed(config['random_seed'])
        logger.info(f"Set random seed to {config['random_seed']}")
    
    # Create output directories
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    volumes_dir = output_path / "volumes"
    masks_dir = output_path / "masks"
    qc_dir = output_path / "qc"
    metrics_dir = output_path / "metrics"
    
    volumes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Find NIfTI files
    nifti_files = list(input_path.glob("*.nii")) + list(input_path.glob("*.nii.gz"))
    if not nifti_files:
        logger.error(f"No NIfTI files found in {input_dir}")
        return
    
    logger.info(f"Found {len(nifti_files)} NIfTI files to process")
    
    # Process each file
    for file_path in nifti_files:
        start_time = time.time()
        logger.info(f"Processing {file_path.name}...")
        
        try:
            # Load volume
            volume, affine, spacing = load_nifti(str(file_path))
            logger.info(f"  Loaded: shape={volume.shape}, spacing={spacing}")
            
            # Store original for visualization
            original_volume = volume.copy()
            original_spacing = spacing
            
            # Ensure finite values
            volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Resample to target spacing
            target_spacing = config.get('target_spacing_mm', 2.0)
            if isinstance(target_spacing, (int, float)):
                target_spacing = (target_spacing, target_spacing, target_spacing)
            
            volume, spacing = resample_isotropic(volume, spacing, target_spacing)
            logger.info(f"  Resampled: shape={volume.shape}, spacing={spacing}")
            
            # Clip intensities
            clip_percentiles = config.get('clip_percentiles', [1.0, 99.0])
            volume = clip_intensities(volume, tuple(clip_percentiles))
            logger.info(f"  Clipped intensities using percentiles {clip_percentiles}")
            
            # Normalize
            normalization = config.get('normalization', 'zscore')
            volume = normalize_volume(volume, normalization)
            logger.info(f"  Normalized using {normalization}")
            
            # Apply denoising filters (Cahier des Charges requirement)
            denoising_cfg = config.get('denoising', {})
            if denoising_cfg.get('enabled', False):
                method = denoising_cfg.get('method', 'gaussian')
                params = denoising_cfg.get('params', {})
                volume = apply_denoising_filters(volume, method, **params)
                logger.info(f"  Applied denoising filter: {method}")
            
            # Apply contrast enhancement (Cahier des Charges requirement)
            contrast_cfg = config.get('contrast_enhancement', {})
            if contrast_cfg.get('enabled', False):
                method = contrast_cfg.get('method', 'clahe')
                params = contrast_cfg.get('params', {})
                volume = apply_contrast_enhancement(volume, method, **params)
                logger.info(f"  Applied contrast enhancement: {method}")
            
            # Optional lung cropping
            mask = None
            if config.get('do_lung_crop', True):
                mask = segment_lungs_simple(volume)
                logger.info(f"  Generated lung mask: {np.sum(mask > 0)} voxels")
                
                # Compute bounding box and crop
                bbox_tight = compute_bbox_from_mask(mask)
                margin_mm = config.get('bbox_margin_mm', 10)
                if isinstance(margin_mm, (int, float)):
                    margin_mm = (margin_mm, margin_mm, margin_mm)
                
                margin_vox = mm_to_vox(margin_mm, spacing)
                bbox = expand_bbox_with_margin(bbox_tight, margin_vox, volume.shape)
                
                volume = volume[bbox]
                mask = mask[bbox]
                affine = update_affine_for_bbox(affine, bbox)
                logger.info(f"  Cropped to bbox with {margin_mm}mm margin")
            
            # Save processed volume
            output_volume_path = volumes_dir / file_path.name
            save_nifti(str(output_volume_path), volume, affine)
            logger.info(f"  Saved volume to {output_volume_path}")
            
            # Save mask if generated
            if mask is not None:
                base_name = file_path.stem.replace('.nii', '')
                output_mask_path = masks_dir / f"{base_name}_mask.nii.gz"
                save_nifti(str(output_mask_path), mask.astype(np.uint8), affine)
                logger.info(f"  Saved mask to {output_mask_path}")
            
            # Generate quality control visualizations
            if config.get('generate_qc_images', True):
                base_name = file_path.stem.replace('.nii', '')
                
                # Create before/after comparison
                qc_comparison_path = qc_dir / f"{base_name}_comparison.png"
                create_qc_visualization(
                    original_volume=original_volume,
                    processed_volume=volume,
                    mask=mask,
                    original_spacing=original_spacing,
                    processed_spacing=spacing,
                    output_path=str(qc_comparison_path)
                )
                
                # Create 3D visualization of processed volume
                qc_3d_path = qc_dir / f"{base_name}_3d_processed.png"
                create_3d_visualization(
                    volume=volume,
                    mask=mask,
                    output_path=str(qc_3d_path),
                    title=f"Processed Volume - {base_name}"
                )
                
                # Create 3D visualization of original volume
                qc_3d_orig_path = qc_dir / f"{base_name}_3d_original.png"
                create_3d_visualization(
                    volume=original_volume,
                    mask=None,
                    output_path=str(qc_3d_orig_path),
                    title=f"Original Volume - {base_name}"
                )
                
                logger.info(f"  Generated QC visualizations in {qc_dir}")
            
            # Calculate and save quality metrics (Cahier des Charges requirement)
            quality_cfg = config.get('quality_metrics', {})
            if quality_cfg.get('enabled', False):
                base_name = file_path.stem.replace('.nii', '')
                
                # Calculate metrics
                metrics = compare_filter_effects(original_volume, volume, "preprocessing_pipeline")
                
                # Save metrics to JSON
                if quality_cfg.get('save_metrics', True):
                    import json
                    metrics_file = metrics_dir / f"{base_name}_metrics.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"  Saved quality metrics to {metrics_file}")
                
                # Log key metrics
                logger.info(f"  Quality metrics - PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.3f}")
            
            elapsed = time.time() - start_time
            logger.info(f"  Completed in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"  ERROR processing {file_path.name}: {e}")
            continue
    
    logger.info("Pipeline completed!")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="3D Lung MRI Preprocessing Pipeline - MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --input data/Task06_Lung/imagesTr --output data/processed
  python pipeline.py --input data/raw --output data/processed --config configs/pipeline.yaml
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directory containing NIfTI files to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Directory to save processed files"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to YAML configuration file (default: configs/pipeline.yaml)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file does not exist: {args.config}")
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
