"""
kmax_fft_accumulator.py

Efficient FFT-based 3D accumulator for maximum sphere intersection order k_max.
Optimized for memory efficiency and numerical robustness.

API:
  estimate_kmax(spheres, config=None)
  verify_exact_kmax(spheres, position=None, config=None)
  batch_estimate(sphere_configs, config=None)
"""

import numpy as np
from scipy import fft as scipy_fft
from typing import Tuple, List, Dict, Optional, Union
import warnings


class FFTConfig:
    """Configuration for FFT accumulator. Memory-conscious defaults."""
    
    def __init__(
        self,
        grid_spacing: float = 0.1,          # Coarser grid = smaller memory
        gaussian_sigma: float = 0.15,       # Broader kernels = more robust
        grid_extent: float = 10.0,          # Smaller extent = smaller grid
        use_gpu: bool = False,
        verify_peak: bool = False,
        epsilon: float = 1e-9,
    ):
        self.grid_spacing = grid_spacing
        self.gaussian_sigma = gaussian_sigma
        self.grid_extent = grid_extent
        self.use_gpu = use_gpu
        self.verify_peak = verify_peak
        self.epsilon = epsilon
        self.grid_size = int(2 * grid_extent / grid_spacing) + 1
    
    def __repr__(self):
        return (
            f"FFTConfig(spacing={self.grid_spacing:.3f}, "
            f"sigma={self.gaussian_sigma:.3f}, size={self.grid_size})"
        )


# ==============================================================================
# Core FFT Accumulator
# ==============================================================================

class FFTAccumulator:
    """FFT-based accumulator for sphere intersection estimation."""
    
    def __init__(self, config: FFTConfig):
        self.cfg = config
        self.use_gpu = config.use_gpu and _has_cupy()
    
    def accumulate(
        self, spheres: List[Dict]
    ) -> Tuple[float, Tuple[float, float, float]]:
        """
        Build accumulator via FFT convolution. Returns only peak info (not full grid).
        
        Args:
          spheres: List of {center: (x, y, z), radius: r}
        
        Returns:
          (peak_value, peak_position):
            - peak_value: Maximum in accumulator (soft k_max estimate)
            - peak_position: (x, y, z) of peak in Ångströms
        """
        if not spheres:
            raise ValueError("Need at least one sphere")
        
        # Create axis once
        ax = np.linspace(-self.cfg.grid_extent, self.cfg.grid_extent, self.cfg.grid_size)
        
        # Create impulse grid
        impulse = np.zeros(
            (self.cfg.grid_size, self.cfg.grid_size, self.cfg.grid_size),
            dtype=np.float32
        )
        
        for sphere in spheres:
            center = np.array(sphere['center'], dtype=np.float32)
            idx = np.round(
                (center + self.cfg.grid_extent) / self.cfg.grid_spacing
            ).astype(int)
            idx = np.clip(idx, 0, self.cfg.grid_size - 1)
            impulse[tuple(idx)] += 1.0
        
        # Convolve with shell kernels for each unique radius
        accumulator = np.zeros_like(impulse)
        unique_radii = sorted(set(float(s['radius']) for s in spheres))
        
        for radius in unique_radii:
            # Create Gaussian shell kernel
            x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            kernel = np.exp(
                -((r_norm - radius)**2) / (2 * self.cfg.gaussian_sigma**2)
            ).astype(np.float32)
            
            # FFT convolution
            accumulator += np.real(scipy_fft.ifftn(
                scipy_fft.fftn(impulse) * scipy_fft.fftn(kernel)
            )).astype(np.float32)
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        peak_value = float(accumulator[peak_idx])
        peak_position = tuple(
            float(i) * self.cfg.grid_spacing - self.cfg.grid_extent
            for i in peak_idx
        )
        
        return peak_value, peak_position


def _has_cupy() -> bool:
    """Check if cupy is available."""
    try:
        import cupy
        return True
    except ImportError:
        return False


# ==============================================================================
# Exact Verification
# ==============================================================================

def _count_intersecting_spheres(
    point: np.ndarray, spheres: List[Dict], epsilon: float
) -> int:
    """Count sphere surfaces passing through point (within epsilon)."""
    count = 0
    for sphere in spheres:
        center = np.array(sphere['center'], dtype=np.float32)
        radius = float(sphere['radius'])
        dist = np.linalg.norm(point - center) - radius
        if abs(dist) <= epsilon:
            count += 1
    return count


# ==============================================================================
# Public API
# ==============================================================================

def estimate_kmax(
    spheres: Union[List[Dict], List[Tuple]],
    config: Optional[FFTConfig] = None,
) -> Tuple[int, float, Tuple]:
    """
    Estimate maximum sphere intersection order via FFT accumulator.
    
    Args:
      spheres: List of sphere definitions:
        - Dict: {center: (x, y, z), radius: r}
        - Tuple: (x, y, z, r)
      config: FFTConfig (defaults shown in class)
    
    Returns:
      (k_max_soft, peak_value, peak_position)
      
      Where:
        k_max_soft: Rounded peak value (soft k_max estimate)
        peak_value: Exact maximum value in accumulator
        peak_position: (x, y, z) location of peak in Ångströms
    """
    if config is None:
        config = FFTConfig()
    
    # Normalize sphere format
    spheres_norm = []
    for s in spheres:
        if isinstance(s, dict):
            spheres_norm.append(s)
        else:
            spheres_norm.append({
                'center': tuple(s[:3]),
                'radius': float(s[3])
            })
    
    # Run accumulator
    acc = FFTAccumulator(config)
    peak_value, peak_position = acc.accumulate(spheres_norm)
    k_max_soft = int(np.round(peak_value))
    
    # Optional exact verification
    if config.verify_peak:
        k_exact = _count_intersecting_spheres(
            np.array(peak_position), spheres_norm, config.epsilon
        )
        k_max_soft = max(k_max_soft, k_exact)
    
    return k_max_soft, peak_value, peak_position


def verify_exact_kmax(
    spheres: Union[List[Dict], List[Tuple]],
    position: Optional[Tuple] = None,
    config: Optional[FFTConfig] = None,
) -> Tuple[int, Dict]:
    """
    Perform exact k_max calculation at a position.
    
    Args:
      spheres: List of sphere definitions
      position: 3D position to check (if None, uses FFT peak)
      config: FFTConfig
    
    Returns:
      (k_max_exact, details_dict)
    """
    if config is None:
        config = FFTConfig()
    
    # Normalize spheres
    spheres_norm = []
    for s in spheres:
        if isinstance(s, dict):
            spheres_norm.append(s)
        else:
            spheres_norm.append({
                'center': tuple(s[:3]),
                'radius': float(s[3])
            })
    
    # If no position, find FFT peak
    if position is None:
        _, _, position = estimate_kmax(spheres_norm, config)
    
    point = np.array(position, dtype=np.float32)
    k_exact = _count_intersecting_spheres(point, spheres_norm, config.epsilon)
    
    details_list = []
    for i, sphere in enumerate(spheres_norm):
        center = np.array(sphere['center'], dtype=np.float32)
        radius = float(sphere['radius'])
        dist = float(np.linalg.norm(point - center) - radius)
        
        details_list.append({
            'idx': i,
            'center': tuple(center),
            'radius': radius,
            'distance': dist,
            'on_surface': abs(dist) <= config.epsilon,
        })
    
    return k_exact, {
        'position': tuple(point),
        'k_exact': k_exact,
        'epsilon': config.epsilon,
        'spheres': details_list,
    }


def batch_estimate(
    sphere_configs: List[Union[List[Dict], List[Tuple]]],
    config: Optional[FFTConfig] = None,
    show_progress: bool = False,
) -> List[Tuple[int, float, Tuple]]:
    """
    Estimate k_max for a batch of sphere configurations.
    
    Args:
      sphere_configs: List of sphere configuration lists
      config: FFTConfig
      show_progress: Print progress
    
    Returns:
      List of (k_max, peak_value, peak_position) tuples
    """
    if config is None:
        config = FFTConfig()
    
    results = []
    for i, spheres in enumerate(sphere_configs):
        if show_progress:
            print(f"  [{i+1}/{len(sphere_configs)}]")
        k_max, peak_val, peak_pos = estimate_kmax(spheres, config)
        results.append((k_max, peak_val, peak_pos))
    
    return results


# ==============================================================================
# Helpers & Testing
# ==============================================================================

def create_test_spheres(
    arrangement: str = 'tetrahedral',
    scale: float = 1.0,
) -> List[Dict]:
    """Create test sphere configurations."""
    
    if arrangement == 'tetrahedral':
        centers = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ]) / np.sqrt(3) * scale
        radii = [1.0] * 4
    
    elif arrangement == 'octahedral':
        centers = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ]) * scale
        radii = [1.0] * 6
    
    elif arrangement == 'cubic':
        centers = np.array([
            [-1, -1, -1], [1, -1, -1],
            [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1],
            [-1, 1, 1], [1, 1, 1],
        ]) * scale / 2
        radii = [0.8] * 8
    
    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")
    
    return [
        {'center': tuple(c), 'radius': float(r)}
        for c, r in zip(centers, radii)
    ]


if __name__ == '__main__':
    print("FFT/Hough Accumulator - Quick Test")
    print("-" * 50)
    
    # Test 1: Tetrahedral
    print("\n[1] Tetrahedral (4 spheres)")
    spheres = create_test_spheres('tetrahedral')
    k, peak, pos = estimate_kmax(spheres)
    print(f"    k_max={k}, peak={peak:.2f}")
    
    # Test 2: Octahedral
    print("\n[2] Octahedral (6 spheres)")
    spheres = create_test_spheres('octahedral')
    k, peak, pos = estimate_kmax(spheres)
    print(f"    k_max={k}, peak={peak:.2f}")
    
    # Test 3: Batch
    print("\n[3] Batch processing")
    configs = [
        create_test_spheres('tetrahedral'),
        create_test_spheres('octahedral'),
        create_test_spheres('cubic'),
    ]
    results = batch_estimate(configs)
    for i, (k, peak, pos) in enumerate(results):
        print(f"    Config {i}: k_max={k}")
    
    print("\nModule ready for integration!")
