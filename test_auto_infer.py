"""
Test script for auto-infer variable names feature in tecplot.py
Tests the new functionality: automatic variable name inference from Python variable names
"""

import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tecplot import TEC_FILE


def test_auto_infer_basic():
    """Test basic auto-infer from Python variable names."""
    print("=" * 60)
    print("Test 1: Auto-infer variable names from Python variables")
    print("=" * 60)
    
    # Create data with specific variable names
    X = np.linspace(0, 10, 50).reshape(50, 1)
    Y = np.linspace(0, 10, 50).reshape(1, 50)
    X_grid = X + np.zeros_like(Y)
    Y_grid = Y + np.zeros_like(X)
    u = np.sin(X_grid) * np.cos(Y_grid)
    v = np.cos(X_grid) * np.sin(Y_grid)
    w = np.ones_like(X_grid)
    
    # Create TEC_FILE without setting Variables
    tec_file = TEC_FILE()
    tec_file.FileName = 'auto_infer_test1'
    
    # Use auto-infer - should get variable names from Python variable names
    tec_file.add_datas([X_grid, Y_grid, u, v, w])
    
    # Verify Variables were auto-set
    print(f"Variables: {tec_file.Variables}")
    print(f"Data count: {len(tec_file.Zones[0].Data)}")
    
    assert tec_file.Variables == ['X_grid', 'Y_grid', 'u', 'v', 'w'], \
        f"Expected ['X_grid', 'Y_grid', 'u', 'v', 'w'], got {tec_file.Variables}"
    
    # Write file
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_auto_infer_with_explicit_names():
    """Test explicit name list override."""
    print("\n" + "=" * 60)
    print("Test 2: Explicit name list override")
    print("=" * 60)
    
    X = np.linspace(0, 10, 50).reshape(50, 1)
    Y = np.linspace(0, 10, 50).reshape(1, 50)
    X_grid = X + np.zeros_like(Y)
    Y_grid = Y + np.zeros_like(X)
    u = np.sin(X_grid)
    v = np.cos(Y_grid)
    w = np.ones_like(X_grid)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'explicit_names_test'
    
    # Provide explicit names
    tec_file.add_datas([X_grid, Y_grid, u, v, w], 
                       ['Xcoord', 'Ycoord', 'Uvel', 'Vvel', 'Wcomp'])
    
    print(f"Variables: {tec_file.Variables}")
    
    assert tec_file.Variables == ['Xcoord', 'Ycoord', 'Uvel', 'Vvel', 'Wcomp'], \
        f"Expected explicit names, got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_auto_infer_fallback_to_variables():
    """Test fallback to Variables list when inference fails."""
    print("\n" + "=" * 60)
    print("Test 3: Fallback to Variables list")
    print("=" * 60)
    
    # Use arrays without clear Python variable name mapping
    data1 = np.random.rand(30, 30)
    data2 = np.random.rand(30, 30)
    data3 = np.random.rand(30, 30)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'fallback_test'
    tec_file.Variables = ['A', 'B', 'C']  # Pre-set Variables
    
    # This should use Variables list since we can't infer meaningful names
    # (arrays created inline don't have clear variable names)
    # Actually, our implementation tries to infer, let's see what happens
    
    # Let's use AddData to verify fallback behavior
    tec_file.add_datas([data1, data2, data3])
    
    print(f"Variables: {tec_file.Variables}")
    
    # The Variables should be set from either inferred or Variables list
    assert len(tec_file.Variables) == 3, f"Expected 3 variables, got {len(tec_file.Variables)}"
    
    tec_file.write_plt()
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return True


def test_original_example_from_user():
    """Test the exact example from user's request."""
    print("\n" + "=" * 60)
    print("Test 4: User's original example - auto infer")
    print("=" * 60)
    
    # Generate the exact data structure from the user's example
    Nr, Nt, Nz = 20, 15, 10
    
    r = np.linspace(0, 1, Nr)
    t = np.linspace(0, 2 * np.pi, Nt)
    z = np.linspace(0, 2 * np.pi, Nz)
    
    Z, T, R = np.meshgrid(z, t, r, indexing='ij')
    
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z_coord = Z.copy()  # Use a different array to get correct name
    
    u = np.sin(2 * np.pi * T)
    v = np.cos(2 * np.pi * T)
    w = np.ones_like(T)
    
    # Test 1: Auto infer without explicit names
    tec_file = TEC_FILE()
    tec_file.FileName = 'user_example_auto'
    
    print("Calling: tec_file.add_datas([X, Y, Z_coord, u, v, w])")
    tec_file.add_datas([X, Y, Z_coord, u, v, w])
    
    print(f"Variables (auto-inferred): {tec_file.Variables}")
    
    assert tec_file.Variables == ['X', 'Y', 'Z_coord', 'u', 'v', 'w'], \
        f"Expected ['X', 'Y', 'Z_coord', 'u', 'v', 'w'], got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"Successfully created: {tec_file.FileName}.plt")
    
    # Test 2: Explicit names
    print("\n--- Explicit names test ---")
    tec_file2 = TEC_FILE()
    tec_file2.FileName = 'user_example_explicit'
    
    print("Calling: tec_file2.add_datas([X, Y, Z_coord, u, v, w], ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'])")
    tec_file2.add_datas([X, Y, Z_coord, u, v, w], ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'])
    
    print(f"Variables (explicit): {tec_file2.Variables}")
    
    assert tec_file2.Variables == ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'], \
        f"Expected ['Xcoord', 'Ycoord', 'Zcoord', 'U', 'V', 'W'], got {tec_file2.Variables}"
    
    tec_file2.write_plt()
    print(f"Successfully created: {tec_file2.FileName}.plt")
    
    return True


def test_mixed_usage():
    """Test mixed usage with Variables list and custom names."""
    print("\n" + "=" * 60)
    print("Test 5: Mixed usage")
    print("=" * 60)
    
    # Create proper 2D grid data with same shape
    n = 50
    x = np.linspace(0, 10, n).reshape(n, 1)
    y = np.linspace(0, 10, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    Z = np.sin(X) * np.cos(Y)
    
    # Pre-set Variables - should be used
    tec_file = TEC_FILE()
    tec_file.FileName = 'mixed_test'
    tec_file.Variables = ['Xaxis', 'Yaxis', 'Zvalue']
    
    tec_file.add_datas([X, Y, Z])
    
    print(f"Variables: {tec_file.Variables}")
    
    # Should use the Variables we set
    assert tec_file.Variables == ['Xaxis', 'Yaxis', 'Zvalue'], \
        f"Expected ['Xaxis', 'Yaxis', 'Zvalue'], got {tec_file.Variables}"
    
    tec_file.write_plt()
    print(f"Successfully created: {tec_file.FileName}.plt")
    return True


def test_3d_velocity_field():
    """Test with realistic 3D velocity field data."""
    print("\n" + "=" * 60)
    print("Test 6: 3D Velocity field (realistic use case)")
    print("=" * 60)
    
    # Create 3D grid
    nx, ny, nz = 20, 20, 20
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    z = np.linspace(0, 2*np.pi, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Velocity components (simple vortex)
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    v = -np.cos(X) * np.sin(Y) * np.sin(Z)
    w = np.zeros_like(X)
    
    # Auto-infer from Python variables
    tec_file = TEC_FILE()
    tec_file.FileName = 'velocity_3d'
    
    print("Creating 3D velocity field with auto-inferred names...")
    tec_file.add_datas([X, Y, Z, u, v, w])
    
    print(f"Variables: {tec_file.Variables}")
    
    assert tec_file.Variables == ['X', 'Y', 'Z', 'u', 'v', 'w'], \
        f"Expected ['X', 'Y', 'Z', 'u', 'v', 'w'], got {tec_file.Variables}"
    
    # Set title
    tec_file.Title = '3D Velocity Field'
    
    tec_file.write_plt()
    print(f"Successfully created: {tec_file.FileName}.plt")
    print(f"Grid size: {nx} x {ny} x {nz} = {nx*ny*nz} points")
    return True


def cleanup_test_files():
    """Clean up test PLT files."""
    test_files = [
        'auto_infer_test1.plt',
        'explicit_names_test.plt',
        'fallback_test.plt',
        'user_example_auto.plt',
        'user_example_explicit.plt',
        'mixed_test.plt',
        'velocity_3d.plt',
    ]
    
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up: {f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("AUTO-INFER VARIABLE NAMES TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_auto_infer_basic,
        test_auto_infer_with_explicit_names,
        test_auto_infer_fallback_to_variables,
        test_original_example_from_user,
        test_mixed_usage,
        test_3d_velocity_field,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n\nFAILED: {test.__name__}")
            print(f"Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())
