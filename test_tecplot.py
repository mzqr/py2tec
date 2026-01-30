"""
Test script for tecplot.py library
Demonstrates creating Tecplot PLT files using TEC_FILE and TEC_ZONE classes
"""

import numpy as np
from tecplot import TEC_FILE, TEC_ZONE


def test_simple_write():
    """Test basic file writing with synthetic data."""
    print("=" * 60)
    print("Test 1: Simple PLT file write")
    print("=" * 60)
    
    # Create TEC_FILE object
    tec_file = TEC_FILE()
    tec_file.FileName = 't_all'
    tec_file.Variables = ['x', 'y', 'z', 'p1', 'p2', 'p3', 'p4']
    
    # Create TEC_ZONE object
    tec_file.Zones = [TEC_ZONE()]
    
    # Create synthetic data (2D grid)
    nx, ny = 100, 100
    x = np.linspace(0, 10, nx).reshape(nx, 1)
    y = np.linspace(0, 10, ny).reshape(1, ny)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    Z = np.sin(X) * np.cos(Y)
    
    # Create pressure fields
    p1 = Z * 1000
    p2 = Z * 2000
    p3 = Z * 3000
    p4 = Z * 4000
    
    # Assign data (must match variable order)
    tec_file.Zones[0].Data = [X, Y, Z, p1, p2, p3, p4]
    tec_file.Zones[0].ZoneName = 'Zone1'
    
    # Write to file
    tec_file.write_plt()
    
    print(f"\nSuccessfully created: {tec_file.FileName}.plt")
    return tec_file


def test_multiple_zones():
    """Test writing multiple zones."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple zones")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'multi_zone'
    tec_file.Variables = ['x', 'y', 'pressure']
    
    # Create 3 zones
    tec_file.Zones = [TEC_ZONE() for _ in range(3)]
    
    for i, zone in enumerate(tec_file.Zones):
        n = 50 + i * 10
        x = np.linspace(i * 10, (i + 1) * 10, n).reshape(n, 1)
        y = np.linspace(0, 10, n).reshape(1, n)
        X = x + np.zeros_like(y)
        Y = y + np.zeros_like(x)
        P = (X + Y) * (i + 1)
        
        zone.Data = [X, Y, P]
        zone.ZoneName = f'Zone_{i}'
        zone.SolutionTime = i * 0.5
    
    # Write to file
    tec_file.write_plt()
    
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with {len(tec_file.Zones)} zones")
    return tec_file


def test_skip_and_begin():
    """Test Skip and Begin parameters."""
    print("\n" + "=" * 60)
    print("Test 3: Skip and Begin parameters")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'skip_test'
    tec_file.Variables = ['x', 'y', 'z']
    
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    # Create high resolution grid
    n = 200
    x = np.linspace(0, 10, n).reshape(n, 1)
    y = np.linspace(0, 10, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    Z = np.sin(X) * np.cos(Y)
    
    zone.Data = [X, Y, Z]
    zone.ZoneName = 'HighResZone'
    zone.Skip = [2, 2, 1]  # Skip every 2 points in x and y
    zone.Begin = [10, 10, 0]  # Skip first 10 points
    zone.EEnd = [10, 10, 0]  # Skip last 10 points
    
    tec_file.write_plt()
    
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with Skip={zone.Skip}")
    return tec_file


def test_3d_data():
    """Test 3D data writing."""
    print("\n" + "=" * 60)
    print("Test 4: 3D data")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'test_3d'
    tec_file.Variables = ['x', 'y', 'z', 'temperature']
    
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    # Create 3D grid
    nx, ny, nz = 30, 30, 30
    x = np.linspace(0, 10, nx).reshape(nx, 1, 1)
    y = np.linspace(0, 10, ny).reshape(1, ny, 1)
    z = np.linspace(0, 10, nz).reshape(1, 1, nz)
    X = x + np.zeros_like(y) + np.zeros_like(z)
    Y = y + np.zeros_like(x) + np.zeros_like(z)
    Z = z + np.zeros_like(x) + np.zeros_like(y)
    T = np.sin(X) * np.cos(Y) * np.sin(Z) + 300  # Temperature in Kelvin
    
    zone.Data = [X, Y, Z, T]
    zone.ZoneName = 'Volume3D'
    
    tec_file.write_plt()
    
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with 3D data")
    return tec_file


def test_different_data_types():
    """Test different data types (int32, float32, etc.)."""
    print("\n" + "=" * 60)
    print("Test 5: Different data types")
    print("=" * 60)
    
    tec_file = TEC_FILE()
    tec_file.FileName = 'data_types'
    tec_file.Variables = ['x', 'y', 'id_int32', 'id_int16', 'id_int8']
    
    tec_file.Zones = [TEC_ZONE()]
    zone = tec_file.Zones[0]
    
    n = 50
    x = np.linspace(0, 10, n, dtype=np.float64).reshape(n, 1)
    y = np.linspace(0, 10, n, dtype=np.float32).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    id32 = np.arange(n * n, dtype=np.int32).reshape(n, n)
    id16 = np.arange(n * n, dtype=np.int16).reshape(n, n)
    # For int8, need to be careful with values > 127
    total = n * n
    values = np.arange(total)
    id8 = (values % 127).astype(np.int8).reshape(n, n)
    
    zone.Data = [X, Y, id32, id16, id8]
    zone.ZoneName = 'MixedTypes'
    
    tec_file.write_plt()
    
    print(f"\nSuccessfully created: {tec_file.FileName}.plt with mixed data types")
    return tec_file


def test_echo_modes():
    """Test different echo modes."""
    print("\n" + "=" * 60)
    print("Test 6: Echo modes")
    print("=" * 60)
    
    # Test with 'full' echo mode
    tec_file = TEC_FILE()
    tec_file.FileName = 'echo_full'
    tec_file.Variables = ['x', 'y', 'value']
    tec_file.set_echo_mode('full', 'full')
    
    tec_file.Zones = [TEC_ZONE()]
    n = 20
    x = np.linspace(0, 1, n).reshape(n, 1)
    y = np.linspace(0, 1, n).reshape(1, n)
    X = x + np.zeros_like(y)
    Y = y + np.zeros_like(x)
    V = X * Y
    
    tec_file.Zones[0].Data = [X, Y, V]
    tec_file.Zones[0].ZoneName = 'TestZone'
    
    tec_file.write_plt()
    print("\n--- Full echo mode completed ---")
    
    # Test with 'none' echo mode
    tec_file2 = TEC_FILE()
    tec_file2.FileName = 'echo_none'
    tec_file2.Variables = ['x', 'y', 'value']
    tec_file2.set_echo_mode('none', 'none')
    
    tec_file2.Zones = [TEC_ZONE()]
    tec_file2.Zones[0].Data = [X, Y, V]
    tec_file2.Zones[0].ZoneName = 'TestZone'
    
    tec_file2.write_plt()
    print("(No output from 'none' echo mode - this is expected)")
    print("\n--- None echo mode completed ---")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TECPLOT.PY TEST SUITE")
    print("=" * 60)
    
    try:
        test_simple_write()
        test_multiple_zones()
        test_skip_and_begin()
        test_3d_data()
        test_different_data_types()
        test_echo_modes()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
