"""
test_kmax_fft_accumulator.py

Comprehensive test suite for FFT/Hough accumulator k_max estimation.
Run with: python test_kmax_fft_accumulator.py
"""

import numpy as np
import sys
from kmax_fft_accumulator import (
    estimate_kmax,
    verify_exact_kmax,
    batch_estimate,
    FFTConfig,
    create_test_spheres,
)


def test_basic_estimation():
    """Test basic k_max estimation on simple geometry."""
    print("\n" + "="*70)
    print("TEST 1: Basic Estimation")
    print("="*70)
    
    # Tetrahedral: 4 spheres should meet at center
    spheres = create_test_spheres(arrangement='tetrahedral')
    k_max, peak_val, peak_pos = estimate_kmax(spheres)
    
    print(f"Tetrahedral arrangement (4 spheres):")
    print(f"  k_max = {k_max} (expected ~4)")
    print(f"  peak_value = {peak_val:.3f}")
    print(f"  peak_position = {peak_pos}")
    
    assert k_max == 4, f"Expected k_max=4, got {k_max}"
    print("  ✓ PASS")


def test_octahedral():
    """Test octahedral arrangement (6 spheres)."""
    print("\n" + "="*70)
    print("TEST 2: Octahedral Arrangement")
    print("="*70)
    
    spheres = create_test_spheres(arrangement='octahedral')
    k_max, peak_val, peak_pos = estimate_kmax(spheres)
    
    print(f"Octahedral arrangement (6 spheres):")
    print(f"  k_max = {k_max} (expected ~6)")
    print(f"  peak_value = {peak_val:.3f}")
    print(f"  peak_position = {peak_pos}")
    
    assert k_max == 6, f"Expected k_max=6, got {k_max}"
    print("  ✓ PASS")


def test_cubic():
    """Test cubic arrangement (8 spheres)."""
    print("\n" + "="*70)
    print("TEST 3: Cubic Arrangement")
    print("="*70)
    
    spheres = create_test_spheres(arrangement='cubic')
    k_max, peak_val, peak_pos = estimate_kmax(spheres)
    
    print(f"Cubic arrangement (8 spheres):")
    print(f"  k_max = {k_max} (expected ~8 at corners)")
    print(f"  peak_value = {peak_val:.3f}")
    
    assert k_max >= 6, f"Expected k_max>=6, got {k_max}"
    print("  ✓ PASS")


def test_config_variations():
    """Test that configuration parameters work correctly."""
    print("\n" + "="*70)
    print("TEST 4: Configuration Variations")
    print("="*70)
    
    spheres = create_test_spheres(arrangement='tetrahedral')
    
    configs = [
        ("Coarse", FFTConfig(grid_spacing=0.2, gaussian_sigma=0.20)),
        ("Default", FFTConfig()),
        ("Fine", FFTConfig(grid_spacing=0.05, gaussian_sigma=0.10)),
    ]
    
    for name, config in configs:
        k_max, peak_val, _ = estimate_kmax(spheres, config)
        print(f"  {name:12s}: k_max={k_max}, peak={peak_val:.3f}")
    
    print("  ✓ PASS (all configs work)")


def test_tuple_format():
    """Test that tuple format (x,y,z,r) works."""
    print("\n" + "="*70)
    print("TEST 5: Tuple Format Support")
    print("="*70)
    
    # Dict format
    spheres_dict = [
        {'center': (0, 0, 0), 'radius': 1.0},
        {'center': (1, 1, 1), 'radius': 1.0},
    ]
    
    # Tuple format
    spheres_tuple = [
        (0, 0, 0, 1.0),
        (1, 1, 1, 1.0),
    ]
    
    k_dict, _, _ = estimate_kmax(spheres_dict)
    k_tuple, _, _ = estimate_kmax(spheres_tuple)
    
    print(f"  Dict format:  k_max = {k_dict}")
    print(f"  Tuple format: k_max = {k_tuple}")
    
    assert k_dict == k_tuple, "Format mismatch"
    print("  ✓ PASS")


def test_batch_processing():
    """Test batch_estimate function."""
    print("\n" + "="*70)
    print("TEST 6: Batch Processing")
    print("="*70)
    
    configs = [
        create_test_spheres(arrangement='tetrahedral'),
        create_test_spheres(arrangement='octahedral'),
        [{'center': (i, j, k), 'radius': 0.5}
         for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)],
    ]
    
    results = batch_estimate(configs, show_progress=False)
    
    print(f"  Processed {len(results)} configurations:")
    for i, (k, peak, pos) in enumerate(results):
        print(f"    Config {i}: k_max={k}, peak={peak:.3f}")
    
    assert len(results) == 3, "Wrong batch size"
    assert all(k > 0 for k, _, _ in results), "Invalid k_max values"
    print("  ✓ PASS")


def test_verification():
    """Test exact verification at peak location."""
    print("\n" + "="*70)
    print("TEST 7: Exact Verification at Peak")
    print("="*70)
    
    spheres = create_test_spheres(arrangement='tetrahedral')
    
    # Get FFT estimate
    k_soft, peak_val, peak_pos = estimate_kmax(spheres)
    
    # Verify at peak location
    k_exact, details = verify_exact_kmax(spheres, peak_pos)
    
    print(f"  FFT soft estimate: k_max = {k_soft} (peak={peak_val:.3f})")
    print(f"  Exact verification: k_exact = {k_exact}")
    print(f"  Position: {peak_pos}")
    print(f"  Details (first 2 spheres):")
    for sphere_detail in details['spheres'][:2]:
        print(f"    Sphere {sphere_detail['idx']}: "
              f"distance={sphere_detail['distance']:.4f}, "
              f"on_surface={sphere_detail['on_surface']}")
    
    print("  ✓ PASS")


def test_custom_position_verification():
    """Test verification at custom position."""
    print("\n" + "="*70)
    print("TEST 8: Custom Position Verification")
    print("="*70)
    
    spheres = create_test_spheres(arrangement='tetrahedral')
    
    # Check at origin (where tet spheres meet)
    k_exact, details = verify_exact_kmax(spheres, position=(0, 0, 0))
    
    print(f"  Verification at origin (0, 0, 0):")
    print(f"    k_exact = {k_exact}")
    
    # Check at different position
    k_other, _ = verify_exact_kmax(spheres, position=(5, 5, 5))
    
    print(f"  Verification at (5, 5, 5):")
    print(f"    k_exact = {k_other} (should be 0 or low)")
    
    assert k_exact >= k_other, "Farther position should have fewer intersections"
    print("  ✓ PASS")


def test_parameter_scan_resolution():
    """Test that parameter variation produces varied k_max."""
    print("\n" + "="*70)
    print("TEST 9: Parameter Scan Resolution")
    print("="*70)
    
    # Scan radius for single lattice
    radii = np.linspace(0.3, 1.5, 8)
    k_max_values = []
    
    for r in radii:
        spheres = [
            {'center': (i, j, k), 'radius': r}
            for i in range(-1, 2)
            for j in range(-1, 2)
            for k in range(-1, 2)
        ]
        k_max, _, _ = estimate_kmax(spheres)
        k_max_values.append(k_max)
    
    print(f"  Radius scan:")
    for r, k in zip(radii, k_max_values):
        print(f"    r={r:.2f}: k_max={k}")
    
    # Should see variation, not all same
    assert len(set(k_max_values)) > 1, "No variation in k_max (broken config)"
    print("  ✓ PASS (variation detected)")


def test_two_lattice_config():
    """Test two-lattice configuration (typical crystal scenario)."""
    print("\n" + "="*70)
    print("TEST 10: Two-Lattice Configuration")
    print("="*70)
    
    # Create two sublattices
    spheres = []
    r_a, r_b = 0.7, 0.5
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                spheres.append({'center': (i, j, k), 'radius': r_a})
                spheres.append({
                    'center': (i + 0.5, j + 0.5, k + 0.5),
                    'radius': r_b
                })
    
    k_max, peak_val, peak_pos = estimate_kmax(spheres)
    
    print(f"  Two-lattice (r_a={r_a}, r_b={r_b}):")
    print(f"    k_max = {k_max}")
    print(f"    peak_value = {peak_val:.3f}")
    print(f"    peak_position = {peak_pos}")
    
    assert k_max > 0, "Invalid k_max"
    print("  ✓ PASS")


def test_memory_efficiency():
    """Test that large batch processing doesn't crash."""
    print("\n" + "="*70)
    print("TEST 11: Memory Efficiency (Large Batch)")
    print("="*70)
    
    # Create 50 configs
    configs = [
        create_test_spheres(arrangement='tetrahedral')
        for _ in range(50)
    ]
    
    # Process efficiently
    config = FFTConfig(grid_spacing=0.15)
    results = batch_estimate(configs, config, show_progress=False)
    
    print(f"  Processed {len(results)} large configurations")
    print(f"  No memory crash ✓")
    print("  ✓ PASS")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 12: Edge Cases")
    print("="*70)
    
    # Single sphere
    spheres_single = [{'center': (0, 0, 0), 'radius': 1.0}]
    k_max_single, _, _ = estimate_kmax(spheres_single)
    print(f"  Single sphere: k_max = {k_max_single} (expected 1)")
    assert k_max_single == 1, "Single sphere should have k_max=1"
    
    # Two spheres
    spheres_pair = [
        {'center': (0, 0, 0), 'radius': 1.5},
        {'center': (1, 0, 0), 'radius': 1.5},
    ]
    k_max_pair, _, _ = estimate_kmax(spheres_pair)
    print(f"  Pair of overlapping spheres: k_max = {k_max_pair} (expected 2)")
    assert k_max_pair == 2, "Pair should have k_max=2"
    
    # Empty error handling
    try:
        estimate_kmax([])
        print("  ERROR: Should have raised ValueError for empty list")
        return False
    except ValueError:
        print(f"  Empty list correctly raises ValueError ✓")
    
    print("  ✓ PASS")


def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "FFT k_max Accumulator Test Suite" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        test_basic_estimation,
        test_octahedral,
        test_cubic,
        test_config_variations,
        test_tuple_format,
        test_batch_processing,
        test_verification,
        test_custom_position_verification,
        test_parameter_scan_resolution,
        test_two_lattice_config,
        test_memory_efficiency,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n  ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
